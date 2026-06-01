import glob
import os
import cv2
import toml

import numpy as np
import pandas as pd

from einops import rearrange
from tqdm import tqdm

from posetail_preprocessing.datasets import BaseDataset
from posetail_preprocessing.utils import io, assemble_extrinsics, top_movement_windows

import re


def true_basename(fname):
    basename = os.path.basename(fname)
    basename = os.path.splitext(basename)[0]
    return basename


def get_cam_name(config, fname):
    basename = true_basename(fname)
    cam_regex = config['triangulation']['cam_regex']
    match = re.search(cam_regex, basename)
    if not match:
        return None
    return match.groups()[0].strip()


class AllenMouseDataset(BaseDataset):

    SUBJECT = 'motor-observatory_717764_2024-12-03_10-47-14'
    TRIAL   = '2024-12-03T10_47_14'

    def __init__(self, dataset_path, dataset_outpath,
                 dataset_name='allen-mouse', error_thresh=None, conf_thresh=0.7):
        super().__init__(dataset_path, dataset_outpath)
        self.dataset_name = dataset_name
        self.error_thresh = error_thresh
        self.conf_thresh  = conf_thresh

    # ------------------------------------------------------------------
    # calibration
    # ------------------------------------------------------------------

    def load_calibration(self, calib_path):

        intrinsics_dict  = {}
        extrinsics_dict  = {}
        distortions_dict = {}
        offset_dict      = {}

        calib_file  = os.path.join(calib_path, 'calibration-videos', 'calibration.toml')
        config_file = os.path.join(calib_path, 'config.toml')

        with open(calib_file, 'r') as f:
            data = toml.load(f)

        with open(config_file, 'r') as f:
            config = toml.load(f)

        for cam, cam_data in data.items():
            if cam == 'metadata':
                continue

            cam_name = cam_data['name']
            offset   = config['cameras'][cam_name]['offset']

            rvec = np.array(cam_data['rotation'])
            tvec = np.array(cam_data['translation'])

            rotation_matrix, _ = cv2.Rodrigues(rvec)
            extrinsics = assemble_extrinsics(rotation_matrix, tvec)

            intrinsics_dict[cam_name]  = cam_data['matrix']
            extrinsics_dict[cam_name]  = extrinsics.tolist()
            distortions_dict[cam_name] = cam_data['distortions']
            offset_dict[cam_name]      = offset[:2]

        return intrinsics_dict, extrinsics_dict, distortions_dict, offset_dict

    # ------------------------------------------------------------------
    # 3-D pose
    # ------------------------------------------------------------------

    def _load_transf_matrix(self, df):
        M = np.identity(3)
        for i in range(3):
            for j in range(3):
                M[i, j] = np.mean(df[f'M_{i}{j}'])
        return M

    def _load_center(self, df):
        center = np.zeros(3)
        for i in range(3):
            center[i] = np.mean(df[f'center_{i}'])
        return center

    def load_pose3d(self, data_path):

        df = pd.read_csv(data_path)

        kpts       = sorted([c for c in df.columns if c.endswith(('_x', '_y', '_z'))])
        unique_kpts = np.unique([k.split('_')[0] for k in kpts])
        error_cols  = [c for c in df.columns if c.endswith('_error')]

        transf_matrix = self._load_transf_matrix(df)
        center        = self._load_center(df)

        coords   = df[kpts].values
        n_frames = coords.shape[0]
        n_kpts   = len(unique_kpts)
        coords   = rearrange(coords, 't (n r) -> t n r', t=n_frames, n=n_kpts)

        if self.error_thresh:
            errors = df[error_cols].values
            errors[np.isnan(errors)] = 10000
            coords[errors >= self.error_thresh] = np.nan

        coords       = rearrange(coords, 't n r -> (t n) r', t=n_frames, n=n_kpts)
        coords_transf = (coords + center).dot(np.linalg.inv(transf_matrix.T))

        pose3d = rearrange(coords_transf, '(t n) r -> 1 t n r', t=n_frames, r=3)
        return {'pose': pose3d, 'keypoints': unique_kpts}

    # ------------------------------------------------------------------
    # 2-D visibility
    # ------------------------------------------------------------------

    def load_vis2d(self, subject_path, trial, keypoints, cam_order):
        """Load per-camera 2D likelihoods and threshold into a bool visibility array.

        Returns shape (1, T, K, V).
        """
        vis_per_cam = []

        for cam in cam_order:
            h5_path = os.path.join(subject_path, 'pose-2d-filtered', f'{cam}_{trial}.h5')
            df = pd.read_hdf(h5_path)

            # MultiIndex columns: (scorer, bodypart, coord)
            # Select likelihood for each keypoint in the canonical 3D order
            scorer = df.columns.get_level_values(0)[0]
            liks   = df.loc[:, (scorer, keypoints, 'likelihood')].values  # (T, K)
            vis_per_cam.append(liks >= self.conf_thresh)

        vis = np.stack(vis_per_cam, axis=-1)        # (T, K, V)
        return vis[np.newaxis]                       # (1, T, K, V)

    # ------------------------------------------------------------------
    # metadata / splits (degenerate — all work is in generate_dataset)
    # ------------------------------------------------------------------

    def generate_metadata(self):
        pass

    def select_splits(self, split_dict=None, split_frames_dict=None, random_state=3):
        self.split_dict        = split_dict        or {}
        self.split_frames_dict = split_frames_dict or {}

    # ------------------------------------------------------------------
    # main generation
    # ------------------------------------------------------------------

    def generate_dataset(self, splits=None):

        subject      = self.SUBJECT
        trial        = self.TRIAL
        subject_path = os.path.join(self.dataset_path, f'{subject}_tracked_v3')

        # --- calibration ---
        intrinsics, extrinsics, distortions, offset_dict = self.load_calibration(subject_path)
        cam_order = list(intrinsics)

        # --- 3D pose ---
        pose_path = os.path.join(subject_path, 'pose-3d', f'{trial}.csv')
        pose_dict = self.load_pose3d(pose_path)
        keypoints = pose_dict['keypoints']

        # --- 2D visibility ---
        pose_dict['vis'] = self.load_vis2d(subject_path, trial, keypoints, cam_order)

        # --- videos ---
        video_dir   = os.path.join(self.dataset_path, subject, 'behavior-videos', 'behavior-videos')
        video_paths = sorted(glob.glob(os.path.join(video_dir, f'*{trial}*.mp4')))

        cam_height_dict = {}
        cam_width_dict  = {}
        fps_list        = []
        nframes_list    = []

        for vp in video_paths:
            cam_name = true_basename(vp).split('_')[0]
            info     = io.get_video_info(vp)
            cam_height_dict[cam_name] = info['camera_heights']
            cam_width_dict[cam_name]  = info['camera_widths']
            fps_list.append(info['fps'])
            nframes_list.append(info['num_frames'])

        nframes = min(
            pose_dict['pose'].shape[1],
            pose_dict['vis'].shape[1],
            min(nframes_list),
        )

        # --- chronological regions ---
        train_end  = int(0.8 * nframes)
        val_start  = train_end + 50
        val_end    = int(0.9 * nframes)
        test_start = int(0.9 * nframes)
        test_end   = nframes

        region = {
            'train': (0,          train_end),
            'val':   (val_start,  val_end),
            'test':  (test_start, test_end),
        }

        if splits is None:
            splits = list(self.split_dict.keys())

        for split in splits:
            n_bouts = self.split_dict.get(split, 1)
            bout    = self.split_frames_dict.get(split, 500)

            r_start, r_end = region[split]

            bout_starts = top_movement_windows(
                pose_dict['pose'], bout, n_bouts,
                frame_start=r_start, frame_end=r_end)

            for start in tqdm(bout_starts, desc=f'{split}'):
                subset = self._subset_pose_dict(dict(pose_dict), start_frame=start, n_frames=bout)

                outpath = os.path.join(
                    self.dataset_outpath, split, 'mouse1', f'{trial}_ix{start}')

                for vp in video_paths:
                    cam_name    = true_basename(vp).split('_')[0]
                    cam_outpath = os.path.join(outpath, 'img', cam_name)
                    os.makedirs(cam_outpath, exist_ok=True)
                    io.save_frames_pyav(vp, start, bout, cam_outpath)

                io.save_npz(subset, outpath, fname='pose3d')

                calib_dict = {
                    'camera_heights':     cam_height_dict,
                    'camera_widths':      cam_width_dict,
                    'num_frames':         bout,
                    'fps':                min(fps_list),
                    'intrinsic_matrices': intrinsics,
                    'extrinsic_matrices': extrinsics,
                    'distortion_matrices':distortions,
                    'offset_dict':        offset_dict,
                    'num_cameras':        len(intrinsics),
                }
                io.save_yaml(data=calib_dict, outpath=outpath, fname='metadata.yaml')
