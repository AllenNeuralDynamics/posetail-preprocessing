"""JohnsonFlyDataset — Johnson-lab fly walking recordings (orthographic rig).

These recordings have orthographic (DLT) cameras and a long, noisy predictor
output. We:
  1. Load the 3D predictions (``data3D.csv``).
  2. Pick clean "bouts" exactly as ``JARVIS-HybridNet/scripts/visualize_bouts.py``
     does (smoothed-confidence + jitter gating).
  3. Crop the first/last ``crop_frames`` of each bout (the ends are particularly
     bad on inspection).
  4. Emit every cropped bout except the last as ``train``; split the last cropped
     bout into ``val`` (first ``val_frames``) and ``test`` (the remainder).

The orthographic DLT calibration is rewritten into the pinhole convention via
``utils.ortho_camera.build_pinhole_cameras`` (see ``preprocess.md``). Frames are
extracted with PyAV (decord misbehaves on these mp4s).
"""
import os
import glob

import numpy as np
import pandas as pd
import polars as pl
from scipy.signal import medfilt
from scipy.ndimage import median_filter
from tqdm import tqdm

from posetail_preprocessing.datasets import BaseDataset
from posetail_preprocessing.utils import io, build_pinhole_cameras


# ---------------------------------------------------------------------------
# Bout detection — copied verbatim from visualize_bouts.py so detection is
# byte-identical to the reference script.
# ---------------------------------------------------------------------------

def compute_jitter(xyz, kernel=7):
    """Per-frame mean Euclidean distance between raw and median-filtered keypoints.

    Returns (T,) array. High values indicate tracking jumps.
    """
    smoothed = median_filter(xyz, size=(kernel, 1, 1))
    dist = np.linalg.norm(xyz - smoothed, axis=2)  # (T, N)
    return np.nanmean(dist, axis=1)                # (T,)


def detect_bouts(conf, xyz, threshold=0.6, min_bout_frames=200,
                 medfilt_kernel=51, jitter_kernel=7, max_bad_fraction=0.2):
    """Find contiguous bouts of high smoothed confidence and low spatial jitter.

    Returns a list of (start, end) tuples (end exclusive), relative to the input.
    """
    mconf = np.nanmean(conf, axis=1)
    mconf_smooth = medfilt(mconf, kernel_size=medfilt_kernel)

    above = mconf_smooth > threshold
    padded = np.concatenate([[False], above, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    candidates = []
    for s, e in zip(starts, ends):
        if (e - s) >= min_bout_frames:
            candidates.append((int(s), int(e)))

    if not candidates:
        return []

    jitter = compute_jitter(xyz, kernel=jitter_kernel)
    all_bout_jitter = np.concatenate([jitter[s:e] for s, e in candidates])
    jitter_thresh = np.nanmedian(all_bout_jitter) * 5.0
    print(f"  Jitter threshold: {jitter_thresh:.3f} "
          f"(5x median of {np.nanmedian(all_bout_jitter):.4f})")

    bouts = []
    for s, e in candidates:
        bad_frac = np.mean(jitter[s:e] > jitter_thresh)
        if bad_frac <= max_bad_fraction:
            bouts.append((s, e))
        else:
            print(f"    Rejecting bout frames {s}-{e}: {bad_frac:.0%} bad frames "
                  f"(>{max_bad_fraction:.0%})")
    return bouts


def true_basename(fname):
    return os.path.splitext(os.path.basename(fname))[0]


class JohnsonFlyDataset(BaseDataset):

    def __init__(self, dataset_path, dataset_outpath, data,
                 dataset_name='johnson-fly', conf_thresh=0.7,
                 gt_conf_thresh=0.7, dlt_scale=10.0,
                 crop_frames=300, val_frames=64):
        super().__init__(dataset_path, dataset_outpath)

        self.dataset_name = dataset_name
        self.data = data                  # list of recording dicts (absolute paths)
        self.conf_thresh = conf_thresh    # pose3d masking threshold
        self.gt_conf_thresh = gt_conf_thresh  # GT-for-calibration threshold
        self.dlt_scale = dlt_scale
        self.crop_frames = crop_frames
        self.val_frames = val_frames

        self.split_frames_dict = None
        self._raw_cache = {}              # recording_idx -> (xyz, conf, keypoints)

    # ------------------------------------------------------------------
    # raw loading
    # ------------------------------------------------------------------

    def _load_raw(self, recording_idx):
        """Load (xyz, conf, keypoints) for ``data[recording_idx]``, subset to
        the [start, start+number] window. Cached on self."""
        if recording_idx in self._raw_cache:
            return self._raw_cache[recording_idx]

        row = self.data[recording_idx]
        pred_dir = row['predictions']
        start = row.get('start', 0)
        number = row.get('number', -1)

        csv_path = os.path.join(pred_dir, 'data3D.csv')
        raw = pl.read_csv(csv_path, has_header=False, infer_schema_length=0)

        # row 0 = joint names (repeated x4), row 1 = x/y/z/confidence labels
        header = raw.row(0)
        n_cols = len(header)
        keypoints = [header[i] for i in range(0, n_cols, 4)]

        data_rows = raw.slice(2)
        data_f = data_rows.with_columns(
            [pl.col(c).cast(pl.Float64, strict=False) for c in data_rows.columns])
        arr = data_f.to_numpy().astype(np.float64)  # (T, 4*N)
        T = arr.shape[0]
        N = n_cols // 4
        arr4 = arr.reshape(T, N, 4)
        xyz = arr4[:, :, :3]
        conf = arr4[:, :, 3]

        # subset to the recording's [start, start+number] window
        if number == -1:
            end = T
        else:
            end = min(T, start + number)
        xyz = xyz[start:end]
        conf = conf[start:end]

        result = (xyz, conf, keypoints)
        self._raw_cache[recording_idx] = result
        return result

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def load_pose3d(self, recording_idx):
        """Build the masked pose dict for a recording (conf < thresh -> NaN)."""
        xyz, conf, keypoints = self._load_raw(recording_idx)

        pose = xyz.copy()
        if self.conf_thresh:
            pose[conf < self.conf_thresh] = np.nan

        pose = pose[np.newaxis, ...]  # (1, T, N, 3)
        return {'pose': pose, 'keypoints': np.asarray(keypoints)}

    def load_calibration(self, recording_idx):
        """Build pinhole cameras for a recording from its ortho DLT calibration.

        GT keypoints (conf > gt_conf_thresh, finite, subsampled) anchor the
        sub-pixel-honest construction.
        """
        row = self.data[recording_idx]
        xyz, conf, _ = self._load_raw(recording_idx)

        finite = np.isfinite(xyz).all(axis=2) & np.isfinite(conf)
        keep = finite & (conf > self.gt_conf_thresh)
        gt = xyz[keep]
        if gt.shape[0] > 200_000:
            idx = np.random.default_rng(0).choice(gt.shape[0], 200_000, replace=False)
            gt = gt[idx]

        intrinsics, extrinsics, distortions, dlt, sizes = build_pinhole_cameras(
            row['calibration'], row['recording'], gt, scale=self.dlt_scale)

        return intrinsics, extrinsics, distortions, dlt, sizes

    def generate_metadata(self):
        """Detect + crop bouts per recording and assign train/val/test rows."""
        rows = []

        for ri, row in enumerate(self.data):
            recording = row['recording']
            start = row.get('start', 0)
            subject = os.path.basename(os.path.dirname(recording))
            trial = os.path.basename(recording)

            xyz, conf, _ = self._load_raw(ri)
            n_cams = len(glob.glob(os.path.join(recording, 'Cam*.mp4')))

            print(f'\n{trial}: detecting bouts over {xyz.shape[0]} frames...')
            bouts = detect_bouts(conf, xyz)
            print(f'  Found {len(bouts)} bouts')

            # crop the noisy ends of each bout
            cropped = []
            for (s, e) in bouts:
                cs, ce = s + self.crop_frames, e - self.crop_frames
                if ce - cs > 0:
                    cropped.append((cs, ce))
            if not cropped:
                print(f'  WARNING: no bouts survive cropping for {trial}, skipping')
                continue

            def make_row(split, local_start, n_frames, bout_ix):
                gs = start + local_start
                return {
                    'id': f'{trial}_bout{bout_ix}_f{gs}-{gs + n_frames}',
                    'session': subject,
                    'subject': subject,
                    'trial': trial,
                    'recording_idx': ri,
                    'bout_idx': bout_ix,
                    'local_start': local_start,
                    'global_start': gs,
                    'n_frames': n_frames,
                    'n_cameras': n_cams,
                    'split': split,
                    'include': True,
                }

            # all but the last cropped bout -> train (all frames)
            for bi, (cs, ce) in enumerate(cropped[:-1]):
                rows.append(make_row('train', cs, ce - cs, bi))

            # last cropped bout -> val (first val_frames) + test (remainder)
            bi = len(cropped) - 1
            cs, ce = cropped[bi]
            bout_len = ce - cs
            val_n = min(self.val_frames, bout_len)
            rows.append(make_row('val', cs, val_n, bi))
            test_n = bout_len - val_n
            if test_n > 0:
                rows.append(make_row('test', cs + val_n, test_n, bi))
            else:
                print(f'  NOTE: last bout for {trial} has only {bout_len} frames '
                      f'(<= val_frames={self.val_frames}); no test row emitted')

        df = pd.DataFrame(rows)
        os.makedirs('metadata', exist_ok=True)
        df.to_csv(os.path.join('metadata', f'metadata_{self.dataset_name}.csv'),
                  index=False)
        self.metadata = df
        return df

    def select_splits(self, split_dict=None, split_frames_dict=None,
                      random_state=3):
        """Splits are fixed by ``generate_metadata`` (bout detection). This just
        stores ``split_frames_dict`` and optionally subsets a split."""
        self.split_frames_dict = split_frames_dict

        if split_dict:
            for split, n in split_dict.items():
                self._select_subset_for_split(split, n=n,
                                              random_state=random_state)

        return self.metadata

    def generate_dataset(self, splits=None):

        valid_splits = np.unique(self.metadata['split'])
        if splits is not None:
            splits = set(splits)
            assert splits.issubset(set(valid_splits))
        else:
            splits = set(valid_splits)

        # group metadata rows by recording so we load + calibrate once each
        for ri, row in enumerate(self.data):
            rec_meta = self.metadata[
                (self.metadata['recording_idx'] == ri)
                & (self.metadata['split'].isin(splits))
                & (self.metadata['include'])]
            if rec_meta.empty:
                continue

            recording = row['recording']
            trial = os.path.basename(recording)
            print(f'\nprocessing {trial} ({len(rec_meta)} bout rows)...')

            # shared across all bouts of this recording
            pose_dict = self.load_pose3d(ri)
            intrinsics, extrinsics, distortions, dlt, sizes = self.load_calibration(ri)

            cam_videos = sorted(glob.glob(os.path.join(recording, 'Cam*.mp4')))
            cam_videos = [v for v in cam_videos if true_basename(v) in intrinsics]

            for _, m in rec_meta.iterrows():
                split = m['split']
                local_start = int(m['local_start'])
                n_frames = int(m['n_frames'])
                bout_ix = int(m['bout_idx'])
                gs = int(m['global_start'])

                trial_dir = f'{trial}_bout{bout_ix}_f{gs}-{gs + n_frames}'
                trial_outpath = os.path.join(
                    self.dataset_outpath, split, m['subject'], trial_dir)
                os.makedirs(trial_outpath, exist_ok=True)

                # 3D pose subset
                pose_subset = self._subset_pose_dict(
                    dict(pose_dict), start_frame=local_start, n_frames=n_frames)
                io.save_npz(pose_subset, trial_outpath, fname='pose3d')

                # frames: PyAV seek-and-decode per camera (global frame indices)
                cam_height_dict = {}
                cam_width_dict = {}
                num_frames = []
                fps = []
                for cam_video_path in tqdm(cam_videos, desc=f'{split}/{trial_dir}'):
                    cam_name = true_basename(cam_video_path)
                    cam_outpath = os.path.join(trial_outpath, 'img', cam_name)
                    video_info = io.save_frames_pyav(
                        cam_video_path, start_frame=gs, n_frames=n_frames,
                        outpath=cam_outpath)
                    cam_height_dict[cam_name] = video_info['camera_heights']
                    cam_width_dict[cam_name] = video_info['camera_widths']
                    num_frames.append(video_info['frames_written'])
                    fps.append(video_info['fps'])

                calib_dict = {
                    'intrinsic_matrices': intrinsics,
                    'extrinsic_matrices': extrinsics,
                    'distortion_matrices': distortions,
                    'dlt_coefficients': dlt,
                    'camera_heights': cam_height_dict,
                    'camera_widths': cam_width_dict,
                    'num_frames': min(num_frames),
                    'fps': min(fps),
                    'num_cameras': len(intrinsics),
                }
                io.save_yaml(data=calib_dict, outpath=trial_outpath,
                             fname='metadata.yaml')
