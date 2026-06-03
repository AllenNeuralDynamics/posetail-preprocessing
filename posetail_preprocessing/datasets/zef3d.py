"""ZefDataset — 3DZeF20 zebrafish 3D pose (two-view: top `T` + front `F`).

Consumes the *pinhole* release at
``/groups/karashchuk/karashchuklab/animal-datasets/3dzef/3DZeF20-pinhole``,
using only its ``train/`` split (the only one with ground-truth points). The
single annotated keypoint per fish is the head (3D world coords, cm).

Splits (sequence + frame-range based)::

    train = ZebraFish-01, -02, -03        (all frames)
    val   = ZebraFish-04, frames [0, 64)
    test  = ZebraFish-04, frames [64, end)

Output (one trial per (split, sequence))::

    {out}/{split}/{ZebraFish-0X}/trial/
        pose3d.npz      {'pose': (S, T, 1, 3) float32, 'keypoints': ['head']}
        metadata.yaml   intrinsics/extrinsics/distortion per cam F & T + dims
        img/F/000000.jpg, 000001.jpg, ...   (copied from imgF, renumbered from 0)
        img/T/000000.jpg, ...

ZebraFish-04 appears under both ``val/`` and ``test/`` with its own frame range;
frames (and the copied images) are renumbered from 0 within each.

Source facts (verified): gt.txt is dense and contiguous — rows == n_fish *
n_frames, frames 1..N with every fish present each frame — so a frame-pivot of the
ground truth aligns 1:1 with the sorted ``img{F,T}/*.jpg`` list. Pinhole calibration
is precomputed: ``cam{F,T}_intrinsic.json`` holds ``K`` (3x3) and ``Distortion``
(1x5); ``cam{F,T}_extrinsic.json`` holds a 4x4 ``extrinsic_matrix`` (world->camera).
No solvePnP is needed.
"""
import glob
import os
import shutil

import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm

from posetail_preprocessing.datasets import BaseDataset
from posetail_preprocessing.utils import io


TRAIN_SEQUENCES = ['ZebraFish-01', 'ZebraFish-02', 'ZebraFish-03']
VALTEST_SEQUENCE = 'ZebraFish-04'
VAL_FRAMES = 64                 # first 64 frames of ZebraFish-04 -> val; rest -> test
CAM_NAMES = ['F', 'T']          # front + top views (imgF / imgT)
KEYPOINTS = ['head']
DEBUG_FRAMES = 16               # frames per job when debug=True


class ZefDataset(BaseDataset):

    def __init__(self, dataset_path, dataset_outpath, dataset_name='3dzef'):
        super().__init__(dataset_path, dataset_outpath)

        self.dataset_name = dataset_name
        # all annotated sequences live under the source `train/` split
        self.source_split_path = os.path.join(dataset_path, 'train')

    def load_calibration(self, session_path):
        """Read precomputed pinhole calibration for cameras F and T.

        Returns three dicts keyed by camera name:
          intrinsics:  3x3 K matrix
          extrinsics:  4x4 world->camera matrix
          distortions: (5, 1) column vector (matches the rest of the pipeline)
        """
        intrinsics_dict = {}
        extrinsics_dict = {}
        distortions_dict = {}

        for cam_name in CAM_NAMES:

            intrinsic = io.load_json(
                os.path.join(session_path, f'cam{cam_name}_intrinsic.json'))
            extrinsic = io.load_json(
                os.path.join(session_path, f'cam{cam_name}_extrinsic.json'))

            intrinsics_dict[cam_name] = np.array(intrinsic['K']).tolist()
            extrinsics_dict[cam_name] = np.array(extrinsic['extrinsic_matrix']).tolist()
            distortions_dict[cam_name] = np.array(intrinsic['Distortion']).T.tolist()

        return intrinsics_dict, extrinsics_dict, distortions_dict

    def load_pose3d(self, data_path):
        """Pivot gt.txt into a dense (n_subjects, n_frames, 1, 3) head-pose array."""

        df = pd.read_csv(data_path, sep=',', header=None)
        df = df.loc[:, 0:4]
        df = df.rename(columns={0: 'frame', 1: 'subject',
                                2: 'head_x', 3: 'head_y', 4: 'head_z'})

        subjects = df['subject'].unique()
        df = df.pivot(index='frame', columns='subject',
                      values=['head_x', 'head_y', 'head_z'])

        # guard: the source is expected to be a contiguous 1..N frame range
        frames = df.index.values
        assert frames[0] == 1 and np.array_equal(frames, np.arange(1, len(frames) + 1)), (
            f'{data_path}: expected contiguous frames 1..N, got '
            f'[{frames[0]}..{frames[-1]}] with {len(frames)} rows')

        columns = [f'sub{subj}_{col}' for col, subj in df.columns]
        df.columns = columns
        pose_df = df[sorted(columns)]
        subject_pose = []

        for subject in subjects:

            sub_df = pose_df[[col for col in df.columns if col.startswith(f'sub{subject}_')]]

            x_cols = sub_df.columns.str.endswith('_x')
            y_cols = sub_df.columns.str.endswith('_y')
            z_cols = sub_df.columns.str.endswith('_z')

            pose_x = np.array(sub_df.loc[:, x_cols].values)
            pose_y = np.array(sub_df.loc[:, y_cols].values)
            pose_z = np.array(sub_df.loc[:, z_cols].values)

            pose3d = np.stack((pose_x, pose_y, pose_z), axis=-1)  # (frame, kpts, 3)
            subject_pose.append(pose3d)

        pose3d = np.stack(subject_pose, axis=0)  # (n_subjects, frame, kpts, 3)

        return {'pose': pose3d.astype(np.float32), 'keypoints': list(KEYPOINTS)}

    def generate_metadata(self):
        """One row per source sequence (train/ only)."""

        sequences = io.get_dirs(self.source_split_path)
        rows = []

        for session in sequences:

            session_path = os.path.join(self.source_split_path, session)
            n_frames = len(glob.glob(os.path.join(session_path, 'imgF', '*.jpg')))

            if session in TRAIN_SEQUENCES:
                split = 'train'
            elif session == VALTEST_SEQUENCE:
                split = 'val+test'
            else:
                split = None

            rows.append({
                'id': session,
                'session': session,
                'subject': session,
                'trial': 1,
                'n_cameras': len(CAM_NAMES),
                'n_frames': n_frames,
                'total_frames': n_frames * len(CAM_NAMES),
                'split': split,
                'include': True})

        os.makedirs('metadata', exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join('metadata', f'metadata_{self.dataset_name}.csv'), index=False)

        self.metadata = df

        return df

    def select_splits(self, split_dict=None, split_frames_dict=None, random_state=3):
        """No-op bookkeeping; authoritative splits are the job list in
        ``generate_dataset``. Signature kept uniform with the other datasets."""
        return self.metadata

    def _jobs(self):
        """(split, session, start, n) tuples; n=None means 'to the end'."""
        jobs = [('train', seq, 0, None) for seq in TRAIN_SEQUENCES]
        jobs.append(('val', VALTEST_SEQUENCE, 0, VAL_FRAMES))
        jobs.append(('test', VALTEST_SEQUENCE, VAL_FRAMES, None))
        return jobs

    def generate_dataset(self, splits=None, debug=False):

        jobs = self._jobs()
        if splits is not None:
            splits = set(splits)
            jobs = [j for j in jobs if j[0] in splits]

        for split, session, start, n in tqdm(jobs, desc='3dzef trials'):

            if debug:
                n = DEBUG_FRAMES

            session_path = os.path.join(self.source_split_path, session)
            outpath = os.path.join(self.dataset_outpath, split, session, 'trial')
            os.makedirs(outpath, exist_ok=True)

            self._process_session(session_path, outpath, start, n)

            if len(os.listdir(outpath)) == 0:
                os.rmdir(outpath)

    def _process_session(self, session_path, outpath, start, n):

        data_path = os.path.join(session_path, 'gt', 'gt.txt')
        if not os.path.isfile(data_path):
            print(f'skipping... could not find {data_path}')
            return

        # load + slice the 3d annotations to this job's frame window
        pose_dict = self.load_pose3d(data_path)
        stop = None if n is None else start + n
        pose_dict['pose'] = pose_dict['pose'][:, start:stop]
        io.save_npz(pose_dict, outpath, fname='pose3d')

        # load calibration
        intrinsics, extrinsics, distortions = self.load_calibration(session_path)

        # copy the matching image window for each camera, renumbered from 0
        cam_height_dict = {}
        cam_width_dict = {}
        n_frames = []

        for cam_name in CAM_NAMES:

            cam_outpath = os.path.join(outpath, 'img', cam_name)
            os.makedirs(cam_outpath, exist_ok=True)

            img_prefix = os.path.join(session_path, f'img{cam_name}')
            img_paths = sorted(glob.glob(os.path.join(img_prefix, '*.jpg')))[start:stop]

            img = cv2.imread(img_paths[0])
            cam_height_dict[cam_name] = img.shape[0]
            cam_width_dict[cam_name] = img.shape[1]
            n_frames.append(len(img_paths))

            for i, img_path in enumerate(img_paths):
                cam_img_outpath = os.path.join(cam_outpath, f'{str(i).zfill(6)}.jpg')
                shutil.copy(img_path, cam_img_outpath)

        cam_dict = {
            'intrinsic_matrices': intrinsics,
            'extrinsic_matrices': extrinsics,
            'distortion_matrices': distortions,
            'camera_heights': cam_height_dict,
            'camera_widths': cam_width_dict,
            'num_frames': min(n_frames),
            'num_cameras': len(intrinsics)}

        io.save_yaml(data=cam_dict, outpath=outpath, fname='metadata.yaml')
