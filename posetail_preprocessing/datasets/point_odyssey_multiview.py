import glob
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd

from einops import rearrange, reduce
from tqdm import tqdm

from posetail_preprocessing.datasets import BaseDataset
from posetail_preprocessing.utils import io


def _load_depth(view_path, frame_idx):
    fname = os.path.join(view_path, 'exr_img', f'depth_{frame_idx + 1:05d}.tiff')
    depth = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f'depth tiff not found or unreadable: {fname}')
    return depth.astype(np.float32)


def _sample_depth_3x3_min(depth, us, vs):
    H, W = depth.shape
    vals = np.full(len(us), np.inf)
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            u_ = np.clip(us + du, 0, W - 1).astype(int)
            v_ = np.clip(vs + dv, 0, H - 1).astype(int)
            vals = np.minimum(vals, depth[v_, u_])
    return vals


def _project_world_to_pixel(P_world, K, E):
    z_cam = (E[:3, :3] @ P_world.T + E[:3, 3:4]).T[:, 2]
    p_cam = (E[:3, :3] @ P_world.T + E[:3, 3:4]).T
    p_img = (K @ p_cam.T).T
    us = p_img[:, 0] / np.where(p_img[:, 2] != 0, p_img[:, 2], 1)
    vs = p_img[:, 1] / np.where(p_img[:, 2] != 0, p_img[:, 2], 1)
    return us, vs, z_cam


class PointOdysseyMultiviewDataset(BaseDataset):

    def __init__(self, dataset_path, dataset_outpath,
                 dataset_name='point_odyssey_multiview'):

        super().__init__(dataset_path, dataset_outpath)
        self.dataset_name = dataset_name

    def load_calibration(self, calib_path):

        cam_dirs = sorted(glob.glob(os.path.join(calib_path, 'view*')))

        intrinsics_dict = {}
        extrinsics_dict = {}
        distortions_dict = {}
        width_dict = {}
        height_dict = {}

        for cam_dir in cam_dirs:
            cam_name = os.path.basename(cam_dir)
            ann = np.load(os.path.join(cam_dir, 'annotations.npz'))

            K = ann['intrinsics'][0].astype(np.float64)
            E = ann['extrinsics'][0].astype(np.float64)

            rgb_files = sorted(glob.glob(os.path.join(cam_dir, 'rgbs', 'rgb_*.jpg')))
            img = cv2.imread(rgb_files[0])
            h, w = img.shape[:2]

            intrinsics_dict[cam_name] = K.tolist()
            extrinsics_dict[cam_name] = E.tolist()
            distortions_dict[cam_name] = np.zeros(5).tolist()
            width_dict[cam_name] = w
            height_dict[cam_name] = h

        return intrinsics_dict, extrinsics_dict, distortions_dict, width_dict, height_dict

    def load_pose3d(self, data_path, eps=1e-6, depth_tol=0.02):

        cam_dirs = sorted(glob.glob(os.path.join(data_path, 'view*')))
        n_views = len(cam_dirs)

        anns = []
        for cam_dir in cam_dirs:
            ann = np.load(os.path.join(cam_dir, 'annotations.npz'))
            anns.append({k: ann[k].astype(np.float64) for k in ann.files})

        # concatenate all views' trajs along the keypoint axis: (T, K_total, 3)
        combined_pose3d = np.concatenate([a['trajs_3d'] for a in anns], axis=1)

        # movement filter first — drops static background points before any depth I/O
        total_movement = reduce(
            np.abs(np.diff(combined_pose3d, axis=0)), 't k r -> k', 'sum'
        )
        movement_check = total_movement > eps
        combined_pose3d = combined_pose3d[:, movement_check, :]

        T, K_filtered, _ = combined_pose3d.shape

        # compute per-view depth-based visibility for the filtered cloud
        vis_pred = np.zeros((T, K_filtered, n_views), dtype=bool)

        for vi, (cam_dir, ann) in enumerate(zip(cam_dirs, anns)):

            # load all T depth tiffs for this view in parallel (I/O bound)
            with ThreadPoolExecutor(max_workers=8) as ex:
                depths = list(ex.map(lambda t: _load_depth(cam_dir, t), range(T)))
            # depths[t] is (H, W) float32

            dh, dw = depths[0].shape

            for t in range(T):
                K = ann['intrinsics'][t]
                E = ann['extrinsics'][t]
                P_t = combined_pose3d[t]

                us, vs, z_cam = _project_world_to_pixel(P_t, K, E)

                valid = (z_cam > 0) & (us >= 0) & (us < dw) & (vs >= 0) & (vs < dh)
                valid_idx = np.where(valid)[0]

                d_sampled = np.full(K_filtered, np.inf)
                if len(valid_idx) > 0:
                    d_sampled[valid_idx] = _sample_depth_3x3_min(
                        depths[t],
                        us[valid_idx].astype(int),
                        vs[valid_idx].astype(int)
                    )

                vis_pred[t, :, vi] = (
                    valid &
                    (z_cam <= d_sampled * (1 + depth_tol)) &
                    (d_sampled < 1e8)
                )

        pose3d = np.expand_dims(combined_pose3d, axis=0)  # (1, T, K, 3)
        vis = np.expand_dims(vis_pred, axis=0)             # (1, T, K, V)

        K_out = pose3d.shape[2]
        keypoints = [f'kpt{i}' for i in range(K_out)]

        return {'pose': pose3d, 'keypoints': keypoints, 'vis': vis}

    def _is_flat_split(self, split_path):
        return os.path.isdir(os.path.join(split_path, 'view0'))

    def _get_sessions(self, split_path, split):
        if self._is_flat_split(split_path):
            return [split], {split: split_path}
        sessions = io.get_dirs(split_path)
        return sessions, {s: os.path.join(split_path, s) for s in sessions}

    def generate_metadata(self):

        rows = []
        splits = [s for s in ('train', 'val', 'test')
                  if os.path.isdir(os.path.join(self.dataset_path, s))]

        for split in splits:
            split_path = os.path.join(self.dataset_path, split)
            sessions, session_paths = self._get_sessions(split_path, split)

            for session in tqdm(sessions, desc='metadata generation'):
                session_path = session_paths[session]
                cams = sorted(glob.glob(os.path.join(session_path, 'view*')))
                if len(cams) == 0:
                    continue
                if not os.path.exists(os.path.join(cams[0], 'annotations.npz')):
                    continue
                metadata_rows = self._get_session(session_path, session, split)
                rows.extend(metadata_rows)

        os.makedirs('metadata', exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join('metadata', f'metadata_{self.dataset_name}.csv'), index=False)
        self.metadata = df
        return df

    def select_splits(self, split_dict=None, split_frames_dict=None,
                      random_state=3):
        self.split_frames_dict = split_frames_dict

        if split_dict:
            for split, n in split_dict.items():
                self._select_subset_for_split(n=n, split=split, random_state=random_state)

        return self.metadata

    def select_train_set(self, n_train_videos=25, seed=3):
        np.random.seed(seed)

        train_ixs = np.random.choice(self.metadata.index, n_train_videos, replace=False)
        train_split = self.metadata.index.isin(train_ixs)

        self.metadata.loc[train_split, 'split'] = 'train'
        self.metadata.loc[train_split, 'include'] = True
        self.metadata.loc[~train_split, 'split'] = 'val'
        self.metadata.loc[~train_split, 'include'] = True

        return self.metadata

    def generate_dataset(self, splits=None):

        valid_splits = pd.unique(self.metadata['split'])

        if splits is not None:
            splits = set(splits)
            assert splits.issubset(valid_splits)
        else:
            splits = valid_splits

        for split in splits:
            split_path = os.path.join(self.dataset_path, split)
            sessions, session_paths = self._get_sessions(split_path, split)

            for session in tqdm(sessions, desc=split):
                session_path = session_paths[session]
                cams = sorted(glob.glob(os.path.join(session_path, 'view*')))
                if len(cams) == 0:
                    continue
                if not os.path.exists(os.path.join(cams[0], 'annotations.npz')):
                    continue

                outpath = os.path.join(self.dataset_outpath, split, session)
                trial_outpath = os.path.join(outpath, 'trial')
                os.makedirs(outpath, exist_ok=True)
                self._process_session(session_path, trial_outpath, session, split)

                if len(os.listdir(outpath)) == 0:
                    os.rmdir(outpath)

    def _get_session(self, session_path, session, split):

        intrinsics_dict, *_ = self.load_calibration(session_path)
        cam_names = list(intrinsics_dict.keys())
        n_cams = len(cam_names)

        img_glob = os.path.join(session_path, cam_names[0], 'rgbs', 'rgb_*.jpg')
        n_frames = len(glob.glob(img_glob))

        rows = [{'id': session,
                 'session': session,
                 'subject': session,
                 'trial': 1,
                 'n_cameras': n_cams,
                 'n_frames': n_frames,
                 'total_frames': n_frames * n_cams,
                 'split': split,
                 'include': True}]

        return rows

    def _process_session(self, session_path, outpath, session, split):

        metadata = self.metadata[self.metadata['split'] == split]
        df = metadata[metadata['id'] == session]
        if df.empty or not df['include'].values[0]:
            return

        split_frames = None
        if self.split_frames_dict and split in self.split_frames_dict:
            split_frames = self.split_frames_dict[split]

        intrinsics, extrinsics, distortions, widths, heights = self.load_calibration(session_path)
        cam_names = list(intrinsics.keys())

        pose_dict = self.load_pose3d(session_path)
        pose_dict = self._subset_pose_dict(pose_dict, n_frames=split_frames)
        io.save_npz(pose_dict, outpath, fname='pose3d')

        n_frames = []

        for cam_name in cam_names:
            img_paths = sorted(glob.glob(
                os.path.join(session_path, cam_name, 'rgbs', 'rgb_*.jpg')
            ))
            img_outpath = os.path.join(outpath, 'img', cam_name)
            os.makedirs(img_outpath, exist_ok=True)
            n_frames.append(len(img_paths))

            for i, img in enumerate(img_paths):
                if split_frames and i == split_frames:
                    break
                new_img_path = os.path.join(img_outpath, f'img{str(i).zfill(6)}.jpg')
                if not os.path.exists(new_img_path):
                    os.symlink(img, new_img_path)

        cam_dict = {
            'intrinsic_matrices': intrinsics,
            'extrinsic_matrices': extrinsics,
            'distortion_matrices': distortions,
            'camera_heights': heights,
            'camera_widths': widths,
            'n_frames': min(n_frames),
            'num_cameras': len(cam_names)
        }

        io.save_yaml(data=cam_dict, outpath=outpath, fname='metadata.yaml')
