import glob
import os 
import cv2
import toml
import shutil

import numpy as np
import pandas as pd 

from einops import rearrange
from tqdm import tqdm

from posetail_preprocessing.datasets import BaseDataset
from posetail_preprocessing.utils import io, assemble_extrinsics


class AniposeFlyDataset(BaseDataset): 

    def __init__(self, dataset_path, dataset_outpath, 
                 dataset_name = 'anipose_fly', error_thresh = None):
        super().__init__(dataset_path, dataset_outpath)

        self.dataset_name = dataset_name
        self.error_thresh = error_thresh

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

        # filter coords
        if self.error_thresh:
            errors = df[error_cols].values
            errors[np.isnan(errors)] = 10000
            error_mask = (errors >= self.error_thresh)
            coords[error_mask] = np.nan

        # undo transformation
        coords = rearrange(coords, 't n r -> (t n) r', t = n_frames, n = n_kpts)
        coords_transf = (coords + center).dot(np.linalg.inv(transf_matrix.T))

        pose3d = rearrange(coords_transf, '(t n) r -> 1 t n r', t = n_frames, r = 3)  # (n_subjects, time, kpts, 3)
        pose3d_dict = {'pose': pose3d, 'keypoints': unique_kpts}

        return pose3d_dict


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


    def select_splits(self, split_dict = None, split_frames_dict = None, 
                      random_state = 3):
        
        self.split_frames_dict = split_frames_dict

        subject_splits = [{'grant/11.29.22/Fly 2_0'},
                          {'grant/11.29.22/Fly 1_0',
                           'grant/11.29.22/Fly 4_0',
                           }]
        splits = ['val', 'test']

        for i, subjects in enumerate(subject_splits):
            self.metadata.loc[self.metadata['subject'].isin(subjects), 'split'] = splits[i]

        # only select 2 validation samples to use
        if split_dict: 
            for split, n in split_dict.items():
                self._select_subset_for_split(split = split, n = n, random_state = random_state)


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

        # select subset of metadata associated with the split 
        metadata = self.metadata[self.metadata['split'] == split]

        calib_path = os.path.dirname(subject_path)
        # load calibration data
        intrinsics, extrinsics, distortions, offset_dict = self.load_calibration(calib_path)

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

            # load and format the 3d annotations
            trial_outpath = os.path.join(outpath, trial)
            os.makedirs(trial_outpath, exist_ok = True)
            print(os.path.join(subject_path, 'pose-3d', f'{trial}*.csv'))
            data_path = glob.glob(os.path.join(subject_path, 'pose-3d', f'{trial}*.csv'))[0]

            pose_dict = self.load_pose3d(data_path)
            pose_dict = self._subset_pose_dict(pose_dict, n_frames = split_frames)

            # load and format the 3d annotations
            pose = pose_dict['pose']
            if split_frames:
                pose_subset = pose[:, :split_frames, :, :]
            else:
                pose_subset = pose

            pose_dict_subset = {'pose': pose_subset, 
                                'keypoints': pose_dict['keypoints']}

            io.save_npz(pose_dict_subset, trial_outpath, fname = 'pose3d')

            if split == 'test':  
                # for test set, save as videos
                video_info = self._process_subject_test(
                    cam_videos, trial_outpath)
            else: 
                # for train and validation sets, deserialize the camera videos 
                # and save as images  
                video_info = self._process_subject_train(
                    cam_videos, trial_outpath, split_frames = split_frames)

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
        

    def _process_subject_train(self, video_paths, trial_outpath, split_frames = None): 

        cam_height_dict = {}
        cam_width_dict = {}
        num_frames = []
        fps = []

        for cam_video_path in video_paths: 
            
            # extract info from the video   
            cam_trial = os.path.splitext(os.path.basename(cam_video_path))[0] 
            # cam_name = cam_trial.split(' ')[2].split('-')[1]
            cam_name = cam_trial.split('Cam-')[1][0]

            # deserialize video into images
            cam_outpath = os.path.join(trial_outpath, 'img', cam_name)
            os.makedirs(cam_outpath, exist_ok = True)

            video_info = io.deserialize_video(
                cam_video_path, 
                cam_outpath, 
                debug_ix = split_frames)
            
            cam_height_dict[cam_name] = video_info['camera_heights']
            cam_width_dict[cam_name] = video_info['camera_widths']
            num_frames.append(video_info['num_frames'])
            fps.append(video_info['fps'])

        video_info = {
            'camera_heights': cam_height_dict, 
            'camera_widths': cam_width_dict, 
            'num_frames': min(num_frames),
            'fps': min(fps)
        }

        return video_info
    
    def _process_subject_test(self, video_paths, trial_outpath): 

        cam_height_dict = {}
        cam_width_dict = {}
        num_frames = []
        fps = []

        outpath = os.path.join(trial_outpath, 'vid')
        os.makedirs(outpath, exist_ok = True)

        for cam_video_path in video_paths: 

            # extract info from the video   
            cam_trial = os.path.splitext(os.path.basename(cam_video_path))[0] 
            # cam_name = cam_trial.split(' ')[2].split('-')[1]
            cam_name = cam_trial.split('Cam-')[1][0]

            video_info = io.get_video_info(cam_video_path)
            cam_height_dict[cam_name] = video_info['camera_heights']
            cam_width_dict[cam_name] = video_info['camera_widths']
            num_frames.append(video_info['num_frames'])
            fps.append(video_info['fps'])

            # copy video to desired location
            cam_video_outpath = os.path.join(outpath, f'{cam_name}.mp4')
            os.symlink(cam_video_path, cam_video_outpath)

        video_info = {
            'camera_heights': cam_height_dict, 
            'camera_widths': cam_width_dict, 
            'num_frames': min(num_frames),
            'fps': min(fps)
        }

        return video_info
