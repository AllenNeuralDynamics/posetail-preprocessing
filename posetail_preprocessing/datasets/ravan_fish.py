"""RavanFishDataset — simulated dense zebrafish 2D pose (single-view, multi-individual).

A simulated zebrafish dataset (from Aniket) of 10 short clips, each a different
crowd of fish (up to 20) tracked at 12 keypoints. We emit the same flat per-trial
layout as the ``rat-city`` / ``chimpact`` siblings (2D-only)::

    {out}/{split}/{video_name}/ix{start}/
        pose2d.npz               key 'pose', shape (S, T, 12, 2), float32, raw px, NaN=occluded/missing
        img/cam0/000000.jpg ...  one JPG per frame; img i <-> pose[:, i]

Splits follow the ``zef3d`` convention (whole videos for train, the held-out video
split by frame range), as requested::

    train = zebrafish_000000 .. _000008   (all 299 frames each)
    val   = zebrafish_000009, frames [0, 32)
    test  = zebrafish_000009, frames [32, end)

Source (``.../test_Lili_dense_fish_max_20``):
  * ``images/train/zebrafish_00000{N}.mp4`` — 640x640, 10 fps, 299 frames.
  * ``labels/train/zebrafish_00000{N}.json`` — COCO-ish, one file per video. A single
    ``videos`` dict, a ``fish`` category, and an ``annotations`` list with **one entry
    per fish** (-> the ``S`` axis). Each annotation carries:
      poses:      list len T, each frame ``[[x0..x11], [y0..y11]]`` (12 kpts, raw px)
      visibility: list T x 12, binary 0.0/1.0 (0 -> not visible, becomes NaN)
"""
import os

import numpy as np
import pandas as pd

from posetail_preprocessing.datasets import BaseDataset
from posetail_preprocessing.utils import io


# 12 keypoints; the source JSON does not name them, so we use generic labels.
KEYPOINTS = [f'kp{k}' for k in range(12)]
N_KEYPOINTS = len(KEYPOINTS)

TRAIN_VIDEOS = [f'zebrafish_{i:06d}' for i in range(9)]   # zebrafish_000000 .. _000008
VALTEST_VIDEO = 'zebrafish_000009'
VAL_FRAMES = 32           # first 32 frames of the held-out video -> val; rest -> test
DEBUG_FRAMES = 16         # frames per job when debug=True


class RavanFishDataset(BaseDataset):

    def __init__(self, dataset_path, dataset_outpath, dataset_name='ravan-fish-sim'):
        super().__init__(dataset_path, dataset_outpath)
        self.dataset_name = dataset_name
        self.labels_dir = os.path.join(dataset_path, 'labels', 'train')
        self.video_dir = os.path.join(dataset_path, 'images', 'train')

    # ------------------------------------------------------------------
    # BaseDataset abstract methods (2D-only -> no 3D / calibration)
    # ------------------------------------------------------------------

    def load_calibration(self, *args, **kwargs):
        return None

    def load_pose3d(self, *args, **kwargs):
        raise NotImplementedError('ravan-fish-sim is a 2D-only dataset; use load_pose2d.')

    # ------------------------------------------------------------------
    # 2D pose loading
    # ------------------------------------------------------------------

    def load_pose2d(self, video_name):
        """Assemble a (S, T, 12, 2) pose array over a video's frames.

        Returns dict with:
          pose:      (S, T, 12, 2) float32, raw px, NaN where visibility == 0.
                     Stored as (x, y).
          keypoints: (12,) joint names
        """
        label_path = os.path.join(self.labels_dir, f'{video_name}.json')
        data = io.load_json(label_path)

        # subjects: one annotation per fish, sorted by id for a stable S axis
        annotations = sorted(data['annotations'], key=lambda a: a['id'])
        S = len(annotations)
        T = max((len(ann['poses']) for ann in annotations), default=0)

        pose = np.full((S, T, N_KEYPOINTS, 2), np.nan, dtype=np.float32)

        for s, ann in enumerate(annotations):
            poses = ann['poses']            # (t, 2, 12): [xs, ys] per frame
            vis = ann['visibility']         # (t, 12)
            for t, (frame_pose, frame_vis) in enumerate(zip(poses, vis)):
                xs, ys = frame_pose
                xy = np.stack([xs, ys], axis=-1).astype(np.float32)  # (12, 2)
                v = np.asarray(frame_vis)
                xy[v == 0] = np.nan         # not visible -> NaN
                pose[s, t] = xy

        return {'pose': pose, 'keypoints': np.asarray(KEYPOINTS)}

    @staticmethod
    def _drop_empty_subjects(pose):
        """Drop subjects (axis 0) that are all-NaN across the trial (cf. chimpact)."""
        keep = ~np.all(np.isnan(pose), axis=(1, 2, 3))
        return pose[keep]

    def _video_split(self, video_name):
        if video_name in TRAIN_VIDEOS:
            return 'train'
        if video_name == VALTEST_VIDEO:
            return 'val+test'
        return None

    # ------------------------------------------------------------------
    # metadata / splits
    # ------------------------------------------------------------------

    def generate_metadata(self):
        """One row per video: split membership, subject / frame counts."""
        video_names = TRAIN_VIDEOS + [VALTEST_VIDEO]
        rows = []
        for video_name in video_names:
            pose = self._drop_empty_subjects(self.load_pose2d(video_name)['pose'])
            rows.append({
                'id': video_name,
                'video_name': video_name,
                'split': self._video_split(video_name),
                'n_subjects': int(pose.shape[0]),
                'n_keyframes': int(pose.shape[1]),
                'include': True,
            })

        df = pd.DataFrame(rows)
        os.makedirs('metadata', exist_ok=True)
        df.to_csv(os.path.join('metadata', f'metadata_{self.dataset_name}.csv'),
                  index=False)
        self.metadata = df
        return df

    def select_splits(self, split_dict=None, split_frames_dict=None, random_state=3):
        """No-op; authoritative splits are the job list in generate_dataset
        (signature kept uniform with the other datasets)."""
        return self.metadata

    # ------------------------------------------------------------------
    # main generation
    # ------------------------------------------------------------------

    def _jobs(self):
        """(split, video_name, start, n) tuples; n=None means 'to the end'."""
        jobs = [('train', v, 0, None) for v in TRAIN_VIDEOS]
        jobs.append(('val', VALTEST_VIDEO, 0, VAL_FRAMES))
        jobs.append(('test', VALTEST_VIDEO, VAL_FRAMES, None))
        return jobs

    def generate_dataset(self, splits=None, debug=False):

        jobs = self._jobs()
        if splits is not None:
            splits = set(splits)
            jobs = [j for j in jobs if j[0] in splits]

        for split, video_name, start, n in jobs:

            if debug:
                n = DEBUG_FRAMES

            pose = self.load_pose2d(video_name)['pose']
            stop = pose.shape[1] if n is None else min(start + n, pose.shape[1])

            pose = self._drop_empty_subjects(pose[:, start:stop])
            if pose.shape[0] == 0 or pose.shape[1] == 0:
                print(f'  skipping {video_name} ({split}): no usable frames')
                continue

            outpath = os.path.join(self.dataset_outpath, split, video_name, f'ix{start}')
            cam_outpath = os.path.join(outpath, 'img', 'cam0')

            video_path = os.path.join(self.video_dir, f'{video_name}.mp4')
            frame_indices = list(range(start, stop))

            print(f'  {split}: {video_name} frames [{start}, {stop}) -> {outpath}')
            info = io.save_frames_decord(video_path, frame_indices, cam_outpath)

            # keep pose aligned with the frames actually written (decord clamps any
            # index >= video length, which can only be the trailing ones)
            written = info.get('frames_written', len(frame_indices))
            if written < pose.shape[1]:
                pose = pose[:, :written]

            io.save_npz({'pose': pose.astype(np.float32)}, outpath, fname='pose2d')
