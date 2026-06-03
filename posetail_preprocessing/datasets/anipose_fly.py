import glob
import os
import cv2
import toml
import shutil
import warnings

import numpy as np
import pandas as pd

from einops import rearrange
from tqdm import tqdm

from posetail_preprocessing.datasets import BaseDataset
from posetail_preprocessing.utils import io, assemble_extrinsics, best_movement_window
from posetail_preprocessing.utils.filtering import mad_filter_coords


class AniposeFlyDataset(BaseDataset):

    def __init__(self, dataset_path, dataset_outpath,
                 dataset_name = 'anipose_fly', error_thresh = None,
                 conf_thresh = 0.7, min_valid_frac = 0.75,
                 max_reproj_error = None, min_ncams = 2, mad_k = 6.0,
                 val_subjects = None, test_subjects = None):
        super().__init__(dataset_path, dataset_outpath)

        self.dataset_name = dataset_name
        # per-keypoint NaN-masking applied in load_pose3d (so the saved 3D pose
        # and the review renders share the exact same surviving points): a point
        # is dropped to NaN when its anipose reprojection error >= error_thresh,
        # its triangulation score < conf_thresh, it used < min_ncams cameras, or
        # it is a temporal MAD outlier (mad_k). Set a threshold to None/0 to skip
        # that stage.
        self.error_thresh = error_thresh
        # 2D likelihood / 3D triangulation-score threshold for keypoint validity
        self.conf_thresh = conf_thresh
        self.min_ncams = min_ncams
        self.mad_k = mad_k
        # quality gates for bout selection
        self.min_valid_frac = min_valid_frac
        self.max_reproj_error = max_reproj_error
        # held-out subjects (chosen from the per-subject quality + visual review):
        # both grant/11.29.22 flies track cleanly and are held out from the
        # sarah-dominated training set
        self.val_subjects = (val_subjects if val_subjects is not None
                             else ['grant/11.29.22/Fly 4_0'])
        self.test_subjects = (test_subjects if test_subjects is not None
                              else ['grant/11.29.22/Fly 1_0'])

    def load_calibration(self, calib_path):

        intrinsics_dict = {}
        extrinsics_dict = {}
        distortions_dict = {}
        offset_dict = {}

        calib_file = os.path.join(calib_path, 'Calibration', 'calibration.toml')
        config_file = os.path.join(calib_path, 'config.toml')

        with open(calib_file, 'r') as f:
            data = toml.load(f)

        with open(config_file, 'r') as f:
            config = toml.load(f)

        cams = list(data.keys())

        for cam in cams: 

            if cam == 'metadata': 
                continue

            cam_data = data[cam]
            cam_name = cam_data['name']
            offset = config['cameras'][cam_name]['offset']

            rvec = np.array(cam_data['rotation'])
            tvec = np.array(cam_data['translation'])

            rotation_matrix, _ = cv2.Rodrigues(rvec)
            extrinsics = assemble_extrinsics(rotation_matrix, tvec)

            intrinsics_dict[cam_name] = cam_data['matrix']
            extrinsics_dict[cam_name] = extrinsics.tolist()
            distortions_dict[cam_name] = cam_data['distortions']
            offset_dict[cam_name] = offset[:2]

        return intrinsics_dict, extrinsics_dict, distortions_dict, offset_dict


    def _load_transf_matrix(self, df):

        transf_matrix = np.identity(3)

        for i in range(3):
            for j in range(3):
                transf_matrix[i, j] = np.mean(df[f'M_{i}{j}'])

        return transf_matrix


    def _load_center(self, df): 

        center = np.zeros(3)

        for i in range(3):
            center[i] = np.mean(df[f'center_{i}'])

        return center


    def load_pose3d(self, data_path):

        df = pd.read_csv(data_path)

        kpts = sorted([col for col in df.columns if col.endswith('_x')
                    or col.endswith('_y') or col.endswith('_z')])
        unique_kpts = np.unique([kpt.split('_')[0] for kpt in kpts])

        error_cols = [col for col in df.columns if col.endswith('_error')]

        # get transformation matrix and center 
        transf_matrix = self._load_transf_matrix(df)
        center = self._load_center(df)

        coords = df[kpts].values
        n_frames = coords.shape[0]
        n_kpts = len(unique_kpts)
        coords = rearrange(coords, 't (n r) -> t n r', t = n_frames, n = n_kpts)

        # per-keypoint anipose quality signals, ordered to match unique_kpts (T, K).
        # error kept raw (un-thresholded) so it can be reused as a quality signal.
        error_arr = np.full((n_frames, n_kpts), np.nan)
        score_arr = np.full((n_frames, n_kpts), np.nan)
        ncams_arr = np.full((n_frames, n_kpts), np.nan)
        for j, kpt in enumerate(unique_kpts):
            if f'{kpt}_error' in df.columns:
                error_arr[:, j] = df[f'{kpt}_error'].values
            if f'{kpt}_score' in df.columns:
                score_arr[:, j] = df[f'{kpt}_score'].values
            if f'{kpt}_ncams' in df.columns:
                ncams_arr[:, j] = df[f'{kpt}_ncams'].values

        # mask unreliable keypoints to NaN so they drop out of the 3D pose (and
        # are not rendered for review). per-keypoint masks are keyed by name, so
        # they stay aligned with the coords keypoint order (unique_kpts).
        drop = np.zeros((n_frames, n_kpts), dtype = bool)
        if self.error_thresh:
            drop |= np.nan_to_num(error_arr, nan = 1e9) >= self.error_thresh
        if self.conf_thresh:
            drop |= np.nan_to_num(score_arr, nan = -1.0) < self.conf_thresh
        if self.min_ncams:
            drop |= np.nan_to_num(ncams_arr, nan = 0.0) < self.min_ncams
        coords[drop] = np.nan

        # undo transformation
        coords = rearrange(coords, 't n r -> (t n) r', t = n_frames, n = n_kpts)
        coords_transf = (coords + center).dot(np.linalg.inv(transf_matrix.T))

        pose3d = rearrange(coords_transf, '(t n) r -> 1 t n r', t = n_frames, r = 3)  # (n_subjects, time, kpts, 3)

        # robust temporal-outlier (MAD) masking, mirroring the dataset cleaning
        if self.mad_k:
            pose3d = mad_filter_coords(pose3d, k = self.mad_k)

        pose3d_dict = {'pose': pose3d, 'keypoints': unique_kpts,
                       'error': error_arr[np.newaxis]}

        return pose3d_dict

    # ------------------------------------------------------------------
    # 2-D visibility
    # ------------------------------------------------------------------

    def load_vis2d(self, subject_path, trial, keypoints, cam_order):
        """Load per-camera 2D likelihoods from pose-2d-filtered and threshold
        into a bool visibility array of shape (1, T, K, V).

        Filenames carry a suffix beyond the trial (``<trial> Cam-<LETTER> ...``),
        so cameras are matched by prefix + ``Cam-<letter>`` rather than an exact
        name. Cameras with no matching h5 contribute an all-False plane.
        """
        keypoints = list(keypoints)
        vis_per_cam = []

        for cam in cam_order:
            pattern = os.path.join(subject_path, 'pose-2d-filtered',
                                   f'{trial}*Cam-{cam}*.h5')
            matches = sorted(glob.glob(pattern))
            if not matches:
                vis_per_cam.append(None)
                continue

            df = pd.read_hdf(matches[0])
            scorer = df.columns.get_level_values(0)[0]
            liks = df.loc[:, (scorer, keypoints, 'likelihood')].values  # (T, K)
            vis_per_cam.append(liks >= self.conf_thresh)

        present = [v for v in vis_per_cam if v is not None]
        if not present:
            return None

        T, K = present[0].shape
        filled = [v if v is not None else np.zeros((T, K), dtype=bool)
                  for v in vis_per_cam]
        vis = np.stack(filled, axis=-1)        # (T, K, V)
        return vis[np.newaxis]                  # (1, T, K, V)


    def generate_metadata(self):
        
        # subjects = io.get_dirs(self.dataset_path)
        os.chdir(self.dataset_path)
        subjects = glob.glob('*/*/Fly *')
        rows = []

        for subject in subjects: 
            subject_path = os.path.join(self.dataset_path, subject)
            metadata_rows = self._get_trials(subject_path, subject)
            rows.extend(metadata_rows)

        os.makedirs('metadata', exist_ok = True)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join('metadata', f'metadata_{self.dataset_name}.csv'), index = False)

        self.metadata = df

        return df


    def _score_bouts(self, splits, split_frames_dict):
        """Score the best movement window + quality per trial for the given splits.

        Writes ``movement_start_frame``, ``movement_score``, ``valid_kpt_frac``
        and ``mean_reproj_error`` columns onto ``self.metadata`` (mirrors the
        3D-POP ``_score_movement_for_splits`` pattern).
        """
        self.metadata['movement_start_frame'] = 0
        self.metadata['movement_score'] = 0.0
        self.metadata['valid_kpt_frac'] = 0.0
        self.metadata['mean_reproj_error'] = np.nan

        rows = self.metadata[self.metadata['split'].isin(splits)]

        for idx, row in tqdm(rows.iterrows(), total = len(rows),
                             desc = f'scoring bouts ({", ".join(splits)})'):

            subject_path = os.path.join(self.dataset_path, row['subject'])
            data_paths = sorted(glob.glob(
                os.path.join(subject_path, 'pose-3d', f"{row['trial']}*.csv")))
            if not data_paths:
                continue

            pose_dict = self.load_pose3d(data_paths[0])
            pose = pose_dict['pose']
            error = pose_dict.get('error')

            w = split_frames_dict.get(row['split'], pose.shape[1])
            start, score = best_movement_window(pose, w)

            win = pose[:, start:start + w]
            finite = np.isfinite(win).all(axis = -1)       # (1, w, K)
            valid_frac = float(np.mean(finite))

            mean_err = np.nan
            if error is not None:
                err_win = error[:, start:start + w]
                if np.isfinite(err_win).any():
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        mean_err = float(np.nanmean(err_win))

            self.metadata.at[idx, 'movement_start_frame'] = start
            self.metadata.at[idx, 'movement_score'] = score
            self.metadata.at[idx, 'valid_kpt_frac'] = valid_frac
            self.metadata.at[idx, 'mean_reproj_error'] = mean_err

    def _apply_quality_filter(self, splits, split_dict = None):
        """Gate bouts per split on valid-keypoint fraction and reprojection error.

        The gate is floor-protected: a split with a positive frame budget is
        never emptied. If the absolute thresholds would leave fewer survivors
        than the split's requested count, the gate is relaxed to keep the
        best-quality bouts up to that count (the visual-review pass is the real
        filter; this just avoids nuking an entire split — see the low-quality
        val subject). Percentiles are logged so thresholds can be tuned.
        """
        split_dict = split_dict or {}

        for split in splits:
            mask = self.metadata['split'] == split
            sub = self.metadata.loc[mask]
            if sub.empty:
                continue

            passes = sub['valid_kpt_frac'] >= self.min_valid_frac
            if self.max_reproj_error is not None:
                passes = passes & ~(sub['mean_reproj_error'] > self.max_reproj_error)

            n_pass = int(passes.sum())

            # floor protection: never drop a budgeted split below what it needs
            n_need = split_dict.get(split)
            floor = n_need if n_need is not None else 1
            floor = min(floor, len(sub))
            if n_pass < floor:
                keep = sub['valid_kpt_frac'].nlargest(floor).index
                passes = pd.Series(sub.index.isin(keep), index=sub.index)
                print(f'[anipose_fly] {split}: quality gate kept {n_pass} bouts '
                      f'(< {floor}); relaxed to top-{int(passes.sum())} by valid_kpt_frac')
                n_pass = int(passes.sum())

            self.metadata.loc[sub.index[~passes], 'include'] = False

            print(f'[anipose_fly] {split}: {n_pass}/{len(sub)} bouts pass quality gate '
                  f'(min_valid_frac={self.min_valid_frac}, '
                  f'max_reproj_error={self.max_reproj_error})')
            for col in ('valid_kpt_frac', 'mean_reproj_error', 'movement_score'):
                q = sub[col].quantile([0.1, 0.5, 0.9]).round(3).to_dict()
                print(f'    {col}: p10/p50/p90 = {q}')

    def _select_top_movement_among_included(self, split, n):
        """Keep the top-n highest-movement bouts among quality-passing rows."""

        split_mask = self.metadata['split'] == split
        survivors = self.metadata[split_mask & self.metadata['include']]

        if n is None or len(survivors) <= n:
            return

        keep_ixs = survivors.nlargest(n, 'movement_score').index
        drop_ixs = survivors.index.difference(keep_ixs)
        self.metadata.loc[drop_ixs, 'include'] = False

    def select_splits(self, split_dict = None, split_frames_dict = None,
                      random_state = 3):

        self.split_frames_dict = split_frames_dict

        # everything starts as train; held-out subjects become val/test
        self.metadata['split'] = 'train'
        self.metadata.loc[self.metadata['subject'].isin(self.val_subjects), 'split'] = 'val'
        self.metadata.loc[self.metadata['subject'].isin(self.test_subjects), 'split'] = 'test'

        # score movement + quality for every split we have a frame budget for,
        # then auto-filter on quality and rank survivors by movement
        score_splits = list(split_frames_dict.keys()) if split_frames_dict else []
        scored = bool(score_splits)
        if scored:
            self._score_bouts(score_splits, split_frames_dict)
            self._apply_quality_filter(score_splits, split_dict = split_dict)

        if split_dict:
            for split, n in split_dict.items():
                if scored and 'movement_score' in self.metadata.columns:
                    self._select_top_movement_among_included(split = split, n = n)
                else:
                    self._select_subset_for_split(split = split, n = n,
                                                  random_state = random_state)

        return self.metadata

    def generate_dataset(self, splits = None): 

        # determine which dataset splits to generate
        valid_splits = np.unique(self.metadata['split'])

        if splits is not None: 
            splits = set(splits)
            assert splits.issubset(valid_splits) 
        else: 
            splits = valid_splits

        os.chdir(self.dataset_path)
        subjects = glob.glob('*/*/Fly *')            
        # generate the dataset for each split
        for split in splits: 
            for subject in tqdm(subjects, desc = split): 
                subject_path = os.path.join(self.dataset_path, subject)
                outpath = os.path.join(self.dataset_outpath, split, subject.replace('/', '-'))
                os.makedirs(outpath, exist_ok = True)
                self._process_subject(subject_path, outpath, split)

                # clean up any empty directories
                if len(os.listdir(outpath)) == 0:
                    os.rmdir(outpath)


    def _get_trials(self, subject_path, subject): 

        calib_path = os.path.join(self.dataset_path, os.path.dirname(subject))
        intrinsics_dict, *_ = self.load_calibration(calib_path)
        n_cams = len(intrinsics_dict)

        video_paths = sorted(glob.glob(os.path.join(subject_path, 'Raw Video', '*.avi')))
        unique_trials = set()
        rows = []

        for i, video_path in enumerate(tqdm(video_paths)):

            trial = os.path.splitext(os.path.basename(video_path))[0]
            # cs = trial.split(' ')
            # trial = f'{cs[0]} {cs[1]}  {cs[3]} {cs[4]}'
            trial = trial.split('Cam')[0].strip()

            cap = cv2.VideoCapture(video_path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if trial not in unique_trials:

                metadata_dict = {
                        'id': trial,
                        'session': subject, 
                        'subject': subject, 
                        'trial': trial,
                        'n_cameras': n_cams, 
                        'n_frames': n_frames,
                        'total_frames': n_frames * n_cams,
                        'split': 'train',
                        'include': True}
            
                unique_trials.add(trial)
                rows.append(metadata_dict)

        return rows
    

    def _process_subject(self, subject_path, outpath, split): 

        # number of images to generate from each video
        split_frames = None
        if self.split_frames_dict and split in self.split_frames_dict: 
            split_frames = self.split_frames_dict[split]

        # select metadata for THIS subject in the split. constraining to the
        # subject is essential: trial ids are not globally unique (every subject
        # has its own dummy_vid_1..5), so matching on id alone would leak one
        # subject's selected bouts into every other subject's split dir.
        subject = os.path.relpath(subject_path, self.dataset_path)
        metadata = self.metadata[(self.metadata['split'] == split)
                                 & (self.metadata['subject'] == subject)]

        calib_path = os.path.dirname(subject_path)
        # load calibration data
        intrinsics, extrinsics, distortions, offset_dict = self.load_calibration(calib_path)
        cam_order = list(intrinsics)

        # get videos
        video_paths = sorted(glob.glob(os.path.join(subject_path, 'Raw Video', f'*.avi')))
        trials = set()

        for i, video_path in enumerate(video_paths): 

            trial = os.path.splitext(os.path.basename(video_path))[0]
            # cs = trial.split(' ')
            # trial = f'{cs[0]} {cs[1]}  {cs[3]} {cs[4]}'
            trial = trial.split('Cam')[0].strip()
            trials.add(trial)

        # traverse the camera names
        for trial in tqdm(trials): 

            # get videos from each camera corresponding to this trial
            # cs = os.path.basename(trial).split(' ')
            # cam_videos = os.path.join(subject_path, 'Raw Video', f'{cs[0]} {cs[1]}*{cs[3]} {cs[4]}.mp4')
            cam_videos = os.path.join(subject_path, 'Raw Video', trial + '*.avi')
            cam_videos = sorted(glob.glob(cam_videos))

            # skip trial if metadata excludes it 
            df = metadata[metadata['id'] == trial]
            if df.empty or not df['include'].values[0]: 
                # print('skipping...')
                continue

            # movement-selected window start for this trial (default 0)
            if 'movement_start_frame' in df.columns and not df.empty:
                start = int(df['movement_start_frame'].values[0])
            else:
                start = 0

            # load and format the 3d annotations
            trial_outpath = os.path.join(outpath, trial)
            os.makedirs(trial_outpath, exist_ok = True)
            data_path = glob.glob(os.path.join(subject_path, 'pose-3d', f'{trial}*.csv'))[0]

            pose_dict = self.load_pose3d(data_path)
            # attach 2D visibility (1, T, K, V) aligned to calibration cam order
            pose_dict['vis'] = self.load_vis2d(
                subject_path, trial, pose_dict['keypoints'], cam_order)
            # 'error' is only a scoring signal; don't persist it
            pose_dict.pop('error', None)
            if pose_dict['vis'] is None:
                pose_dict.pop('vis')

            # window pose (and vis) to the selected movement bout
            pose_dict = self._subset_pose_dict(
                pose_dict, start_frame = start, n_frames = split_frames)

            io.save_npz(pose_dict, trial_outpath, fname = 'pose3d')

            # extract the matching window of frames for every camera (test too)
            video_info = self._process_subject_train(
                cam_videos, trial_outpath,
                split_frames = split_frames, start_frame = start)

            calib_dict = {
                'intrinsic_matrices': intrinsics, 
                'extrinsic_matrices': extrinsics, 
                'distortion_matrices': distortions,
                'offset_dict': offset_dict,
                'num_cameras': len(intrinsics)
            }
            calib_dict.update(video_info)

            # save camera metadata
            io.save_yaml(data = calib_dict, outpath = trial_outpath, 
                    fname = 'metadata.yaml')
        

    def _process_subject_train(self, video_paths, trial_outpath,
                               split_frames = None, start_frame = 0):

        cam_height_dict = {}
        cam_width_dict = {}
        num_frames = []
        fps = []

        for cam_video_path in video_paths:

            # extract info from the video
            cam_trial = os.path.splitext(os.path.basename(cam_video_path))[0]
            cam_name = cam_trial.split('Cam-')[1][0]

            # extract the selected window of frames as images
            cam_outpath = os.path.join(trial_outpath, 'img', cam_name)
            os.makedirs(cam_outpath, exist_ok = True)

            if split_frames:
                # frame-accurate seek-and-decode for the movement window
                video_info = io.save_frames_pyav(
                    cam_video_path, start_frame, split_frames, cam_outpath)
                n = video_info.get('frames_written', split_frames)
            else:
                video_info = io.deserialize_video(cam_video_path, cam_outpath)
                n = video_info['num_frames']

            cam_height_dict[cam_name] = video_info['camera_heights']
            cam_width_dict[cam_name] = video_info['camera_widths']
            num_frames.append(n)
            fps.append(video_info['fps'])

        video_info = {
            'camera_heights': cam_height_dict,
            'camera_widths': cam_width_dict,
            'num_frames': min(num_frames),
            'fps': min(fps)
        }

        return video_info
