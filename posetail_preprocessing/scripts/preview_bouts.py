"""Render multi-camera overlay montages for anipose-fly candidate bouts.

For each bout this picks a few frames spread across the selected movement
window, reprojects the 3D keypoints onto a handful of cameras, and tiles the
results into a single ``cameras x frames`` montage. The montages let a human (or
model) eyeball tracking quality.

Two modes:

  * ``--by-subject`` (exploratory): score EVERY trial across all subjects as one
    pool, write a per-subject quantitative summary, and render the top bouts per
    subject so a val/test subject can be chosen from the quantitative + visual
    checks. Output:
        <outdir>/subject_summary.csv           one row per subject
        <outdir>/<subject>/<trial>.jpg         top-K montages per subject
        <outdir>/index.csv                     one row per rendered bout

  * default (split-based): use the dataset's hardcoded val/test assignment +
    quality gate, render the included bouts. Output:
        <outdir>/<split>/<subject>__<trial>.jpg
        <outdir>/index.csv

Run as a module:
  pixi run python -m posetail_preprocessing.scripts.preview_bouts --by-subject \\
      --bouts-per-subject 4 --n-frames 5 --cameras 3
"""
import argparse
import glob
import os

import cv2
import numpy as np
import pandas as pd

from posetail_preprocessing.datasets import AniposeFlyDataset
from posetail_preprocessing.utils import io, project_points, draw_keypoints, make_montage


DEFAULT_DATASET_PATH = '/groups/karashchuk/karashchuklab/animal-datasets/tuthill-fly'
DEFAULT_SPLIT_FRAMES = {'train': 60, 'val': 16, 'test': 16}


def _video_for_cam(subject_path, trial, cam):
    matches = sorted(glob.glob(
        os.path.join(subject_path, 'Raw Video', f'{trial}*Cam-{cam}*.avi')))
    return matches[0] if matches else None


def _render_bout(dataset, row, n_frames, n_cams, split_frames, cam_names=None):
    """Build one cameras x frames montage for a single bout row. Returns the
    montage image (BGR) or None if it couldn't be rendered. Camera order is
    derived from the subject's own calibration (handles the different rigs).

    cam_names: explicit list of camera names to render (e.g. ['A','D','F'] for
    left/top/right). Names absent from this subject's rig are skipped. When None,
    falls back to the first ``n_cams`` cameras in calibration order."""

    subject = row['subject']
    trial = row['trial']
    subject_path = os.path.join(dataset.dataset_path, subject)
    calib_path = os.path.dirname(subject_path)

    intrinsics, extrinsics, distortions, offset_dict = dataset.load_calibration(calib_path)
    cam_order = list(intrinsics)

    pose_paths = sorted(glob.glob(
        os.path.join(subject_path, 'pose-3d', f'{trial}*.csv')))
    if not pose_paths:
        return None

    pose_dict = dataset.load_pose3d(pose_paths[0])
    pose = pose_dict['pose']                       # (1, T, K, 3)
    keypoints = pose_dict['keypoints']
    vis = dataset.load_vis2d(subject_path, trial, keypoints, cam_order)  # (1,T,K,V) or None

    T = pose.shape[1]
    start = int(row.get('movement_start_frame', 0))
    w = min(split_frames, T - start) if split_frames else T - start
    if w <= 0:
        return None

    # frames spread evenly across the window
    frame_ixs = np.unique(
        np.linspace(start, start + w - 1, n_frames).astype(int))

    if cam_names:
        cams = [c for c in cam_names if c in cam_order]
    else:
        cams = cam_order[:n_cams]
    tiles = []

    for cam in cams:
        v = cam_order.index(cam)
        video_path = _video_for_cam(subject_path, trial, cam)

        for f in frame_ixs:
            if video_path is None:
                tiles.append(None)
                continue
            frame = io.get_frame_synced(video_path, int(f))
            if frame is None:
                tiles.append(None)
                continue

            X = pose[0, f]                          # (K, 3)
            pts2d = project_points(
                intrinsics[cam], distortions[cam], extrinsics[cam], X,
                offset=offset_dict.get(cam))
            cam_vis = vis[0, f, :, v] if vis is not None else None
            tiles.append(draw_keypoints(frame, pts2d, vis=cam_vis))

    return make_montage(tiles, rows=len(cams), cols=len(frame_ixs))


def _index_row(row, split, out_path):
    return {
        'id': row['trial'],
        'subject': row['subject'],
        'split': split,
        'preview_path': out_path,
        'movement_start_frame': int(row.get('movement_start_frame', 0)),
        'movement_score': float(row.get('movement_score', np.nan)),
        'mean_reproj_error': float(row.get('mean_reproj_error', np.nan)),
        'valid_kpt_frac': float(row.get('valid_kpt_frac', np.nan)),
    }


def run_by_subject(dataset, args):
    """Score every trial as one pool, summarize per subject, render top bouts."""

    bout_frames = args.bout_frames

    dataset.generate_metadata()
    meta = dataset.get_metadata()

    # treat everything as one pool so scoring is independent of any split choice
    meta['split'] = 'train'
    dataset.set_metadata(meta)
    dataset._score_bouts(['train'], {'train': bout_frames})
    meta = dataset.get_metadata()

    # ---- per-subject quantitative summary ----
    passing = meta['valid_kpt_frac'] >= args.min_valid_frac
    meta = meta.assign(_pass=passing)
    summary = (meta.groupby('subject')
               .agg(n_trials=('trial', 'count'),
                    n_pass=('_pass', 'sum'),
                    valid_frac_med=('valid_kpt_frac', 'median'),
                    valid_frac_p90=('valid_kpt_frac', lambda s: s.quantile(0.9)),
                    reproj_med=('mean_reproj_error', 'median'),
                    movement_med=('movement_score', 'median'),
                    movement_max=('movement_score', 'max'))
               .round(3)
               .sort_values('valid_frac_med', ascending=False)
               .reset_index())

    os.makedirs(args.outdir, exist_ok=True)
    summary_path = os.path.join(args.outdir, 'subject_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f'\n[preview] per-subject summary -> {summary_path}')
    print(summary.to_string(index=False))

    # ---- render top-K bouts per subject ----
    index_rows = []

    for subject, sdf in meta.groupby('subject'):
        pool = sdf[sdf['valid_kpt_frac'] >= args.min_valid_frac]
        if len(pool) < args.bouts_per_subject:
            print(f'[preview] {subject}: only {len(pool)} bouts pass quality; '
                  f'filling from best-quality remainder')
            pool = sdf.nlargest(
                max(args.bouts_per_subject, len(pool)), 'valid_kpt_frac')
        chosen = pool.nlargest(args.bouts_per_subject, 'movement_score')

        subj_dir = os.path.join(args.outdir, subject.replace('/', '-'))
        os.makedirs(subj_dir, exist_ok=True)

        for _, row in chosen.iterrows():
            montage = _render_bout(
                dataset, row, args.n_frames, args.cameras, bout_frames,
                cam_names=args.camera_names)
            if montage is None:
                print(f'[preview] skipped (no frames): {subject} / {row["trial"]}')
                continue
            out_path = os.path.join(subj_dir, f"{row['trial']}.jpg")
            cv2.imwrite(out_path, montage)
            index_rows.append(_index_row(row, 'unassigned', out_path))
            print(f'[preview] wrote {out_path}')

    _write_index(args.outdir, index_rows)


def run_by_split(dataset, args):
    """Use the dataset's hardcoded val/test assignment + quality gate."""

    split_frames = dict(DEFAULT_SPLIT_FRAMES)

    dataset.generate_metadata()
    split_dict = {s: None for s in split_frames}
    dataset.select_splits(split_dict=split_dict, split_frames_dict=split_frames)

    meta = dataset.get_metadata()
    candidates = meta[meta['include']]
    if args.split:
        candidates = candidates[candidates['split'] == args.split]

    # deterministic order so --shard slices are disjoint and stable across runs
    candidates = candidates.sort_values(['split', 'subject', 'trial'])
    if args.num_shards > 1:
        candidates = candidates.iloc[args.shard::args.num_shards]
        print(f'[preview] shard {args.shard}/{args.num_shards}: '
              f'{len(candidates)} bouts')

    index_rows = []
    for split, split_df in candidates.groupby('split'):
        if args.limit is not None and len(split_df) > args.limit:
            print(f'[preview] {split}: limiting {len(split_df)} -> {args.limit} bouts')
            split_df = split_df.head(args.limit)

        split_outdir = os.path.join(args.outdir, split)
        os.makedirs(split_outdir, exist_ok=True)

        for _, row in split_df.iterrows():
            montage = _render_bout(
                dataset, row, args.n_frames, args.cameras, split_frames.get(split),
                cam_names=args.camera_names)
            if montage is None:
                print(f'[preview] skipped (no frames): {row["subject"]} / {row["trial"]}')
                continue
            fname = f"{row['subject'].replace('/', '-')}__{row['trial']}.jpg"
            out_path = os.path.join(split_outdir, fname)
            cv2.imwrite(out_path, montage)
            index_rows.append(_index_row(row, split, out_path))
            print(f'[preview] wrote {out_path}')

    index_name = 'index.csv' if args.num_shards == 1 else f'index_shard{args.shard}.csv'
    _write_index(args.outdir, index_rows, index_name=index_name)


def _write_index(outdir, index_rows, index_name='index.csv'):
    if index_rows:
        os.makedirs(outdir, exist_ok=True)
        index_path = os.path.join(outdir, index_name)
        pd.DataFrame(index_rows).to_csv(index_path, index=False)
        print(f'[preview] {len(index_rows)} montages -> {index_path}')
    else:
        print('[preview] no montages rendered')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset-path', default=DEFAULT_DATASET_PATH)
    parser.add_argument('--dataset-name', default='anipose_fly')
    parser.add_argument('--outdir', default='preview')
    parser.add_argument('--by-subject', action='store_true',
                        help='score all subjects as one pool and render top bouts '
                             'per subject (for choosing val/test subjects)')
    parser.add_argument('--bouts-per-subject', type=int, default=4,
                        help='montages per subject in --by-subject mode')
    parser.add_argument('--bout-frames', type=int, default=60,
                        help='movement-window length used for scoring in --by-subject mode')
    parser.add_argument('--split', default=None,
                        help='split-based mode: only render this split')
    parser.add_argument('--n-frames', type=int, default=5,
                        help='frames sampled across each bout window')
    parser.add_argument('--cameras', type=int, default=3,
                        help='number of cameras per montage (first N in calib order)')
    parser.add_argument('--camera-names', type=lambda s: s.split(','), default=None,
                        help='explicit comma-separated camera names to render, e.g. '
                             'A,D,F (left,top,right). Overrides --cameras.')
    parser.add_argument('--limit', type=int, default=None,
                        help='split-based mode: cap bouts rendered per split')
    parser.add_argument('--shard', type=int, default=0,
                        help='split-based mode: this shard index (0..num-shards-1)')
    parser.add_argument('--num-shards', type=int, default=1,
                        help='split-based mode: total shards for parallel rendering')
    parser.add_argument('--error-thresh', type=float, default=5.0)
    parser.add_argument('--conf-thresh', type=float, default=0.7)
    parser.add_argument('--min-valid-frac', type=float, default=0.75)
    parser.add_argument('--max-reproj-error', type=float, default=None)
    parser.add_argument('--min-ncams', type=int, default=2,
                        help='drop 3D keypoints triangulated from < this many cameras')
    parser.add_argument('--mad-k', type=float, default=6.0,
                        help='temporal MAD outlier threshold (robust stds); 0 to disable')
    args = parser.parse_args()

    dataset = AniposeFlyDataset(
        dataset_path=args.dataset_path,
        dataset_outpath=args.outdir,
        dataset_name=args.dataset_name,
        error_thresh=args.error_thresh,
        conf_thresh=args.conf_thresh,
        min_valid_frac=args.min_valid_frac,
        max_reproj_error=args.max_reproj_error,
        min_ncams=args.min_ncams,
        mad_k=(args.mad_k if args.mad_k else None))

    if args.by_subject:
        run_by_subject(dataset, args)
    else:
        run_by_split(dataset, args)


if __name__ == '__main__':
    main()
