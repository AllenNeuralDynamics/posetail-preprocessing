"""RatCityDataset — Branson "rat city" 2D pose (single-view, multi-individual).

A single continuous arena recording with 12 rats (6 per side), tracked at 4
keypoints (nose, left ear, right ear, tail base) with no ID switches. We emit the
same flat per-trial layout as the ``chimpact`` sibling (2D-only)::

    {out}/{split}/{clip_name}/ix{start}/
        pose2d.npz               key 'pose', shape (S, T, 4, 2), float32, raw px, NaN=missing
        img/cam0/000000.jpg ...  one JPG per frame; img i <-> pose[:, i]

Because the source is one long recording (not many short clips), the structure
mirrors ``johnson_mouse`` rather than ``chimpact``: contiguous chronological splits
computed in ``generate_dataset``, with ``generate_metadata`` / ``select_splits`` as
no-ops.

Source (``/groups/branson/bransonlab/manan/cohort7_20251209_1659``):
  * ``movie.avi`` — 4696x2048, 40 fps, ~71,993 frames.
  * ``solution_keypoints.csv`` — space-delimited. Columns:
      group id t y x parent_id tracklet_id original_id original_tracklet_id
      kp0_y kp0_x kp1_y kp1_x kp2_y kp2_x kp3_y kp3_x
    ``t`` is the frame index, ``tracklet_id`` in [1, 12] is the persistent animal id
    (-> the ``S`` axis), and ``kp{0..3}_{y,x}`` are the 4 keypoints in (y, x) order.
    Not every (animal, frame) pair has a row; gaps become NaN.
"""
import os

import numpy as np
import pandas as pd

from posetail_preprocessing.datasets import BaseDataset
from posetail_preprocessing.utils import io


# 4 keypoints, in the order stored as kp0..kp3 in solution_keypoints.csv.
KEYPOINTS = ['nose', 'left_ear', 'right_ear', 'tail_base']

N_TRACKLETS = 12          # 12 rats, tracklet_id in [1, 12]
CLIP_NAME = 'cohort7_20251209_1659'


class RatCityDataset(BaseDataset):

    def __init__(self, dataset_path, dataset_outpath, dataset_name='rat-city'):
        super().__init__(dataset_path, dataset_outpath)
        self.dataset_name = dataset_name
        self.csv_path = os.path.join(dataset_path, 'solution_keypoints.csv')
        self.video_path = os.path.join(dataset_path, 'movie.avi')

    # ------------------------------------------------------------------
    # BaseDataset abstract methods (2D-only -> no 3D / calibration)
    # ------------------------------------------------------------------

    def load_calibration(self, *args, **kwargs):
        return None

    def load_pose3d(self, *args, **kwargs):
        raise NotImplementedError('rat-city is a 2D-only dataset; use load_pose2d.')

    # ------------------------------------------------------------------
    # 2D pose loading
    # ------------------------------------------------------------------

    def load_pose2d(self):
        """Assemble a (S, T, 4, 2) pose array over the whole recording.

        Returns dict with:
          pose:      (S, T, 4, 2) float32, raw px, NaN where a tracklet has no
                     detection in that frame. Stored as (x, y).
          keypoints: (4,) joint names
        """
        df = pd.read_csv(self.csv_path, sep=r'\s+')

        S = N_TRACKLETS
        T = int(df['t'].max()) + 1
        K = len(KEYPOINTS)

        pose = np.full((S, T, K, 2), np.nan, dtype=np.float32)

        s = df['tracklet_id'].values.astype(int) - 1   # tracklet_id 1..12 -> 0..11
        t = df['t'].values.astype(int)

        for k in range(K):
            # source is (y, x); store as (x, y) to match chimpact
            pose[s, t, k, 0] = df[f'kp{k}_x'].values
            pose[s, t, k, 1] = df[f'kp{k}_y'].values

        return {'pose': pose, 'keypoints': np.asarray(KEYPOINTS)}

    # ------------------------------------------------------------------
    # metadata / splits (degenerate — all work is in generate_dataset)
    # ------------------------------------------------------------------

    def generate_metadata(self):
        pass

    def select_splits(self, split_dict=None, split_frames_dict=None, random_state=3):
        pass  # everything happens in generate_dataset

    # ------------------------------------------------------------------
    # main generation
    # ------------------------------------------------------------------

    def generate_dataset(self, splits=None, debug=False):

        pose_dict = self.load_pose2d()

        info = io.get_video_info(self.video_path)
        nframes = min(int(info['num_frames']), pose_dict['pose'].shape[1])

        train_end = int(0.8 * nframes)

        # chronological 80/10/10 windows: train = full first 80% as one bout,
        # val = 64 frames just after, test = last 500 frames.
        regions = {
            'train': (0, train_end if not debug else 64),
            'val':   (train_end + 50, 64),
            'test':  (nframes - 500, 500),
        }

        if splits is None:
            splits = set(regions)
        else:
            splits = set(splits)

        for split in splits:
            start, n = regions[split]

            subset = self._subset_pose_dict(dict(pose_dict), start_frame=start, n_frames=n)

            outpath = os.path.join(
                self.dataset_outpath, split, CLIP_NAME, f'ix{start}')
            cam_outpath = os.path.join(outpath, 'img', 'cam0')

            print(f'  {split}: frames [{start}, {start + n}) -> {outpath}')
            winfo = io.save_frames_pyav(self.video_path, start, n, cam_outpath,
                                        progress=True, desc=f'{split} frames')

            # keep pose aligned with the frames actually written
            written = winfo.get('frames_written', n)
            pose = subset['pose']
            if written < pose.shape[1]:
                pose = pose[:, :written]

            io.save_npz({'pose': pose.astype(np.float32)}, outpath, fname='pose2d')
