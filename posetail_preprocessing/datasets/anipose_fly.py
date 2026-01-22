import glob
import os 
import cv2
import toml
import shutil

import numpy as np
import pandas as pd 

from collections import defaultdict

from posetail_preprocessing.datasets import BaseDataset
from posetail_preprocessing.utils import io, assemble_extrinsics


class AniposeFlyDataset(BaseDataset): 

    def __init__(self, dataset_path, dataset_outpath, 
                 dataset_name = 'anipose_fly', 
                 debug_ix = None):
        super().__init__(dataset_path, dataset_outpath)

        self.dataset_name = dataset_name
        self.metadata = None
        self.debug_ix = debug_ix

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

    def load_pose3d(self, data_path):

        df = pd.read_csv(data_path)

        kpts = [col for col in df.columns if col.endswith('_x')
                or col.endswith('_y') or col.endswith('_z')]

        unique_kpts = np.unique([kpt.split('_')[0] for kpt in kpts])

        coords = df[kpts].values
        n_frames, _ = coords.shape
        pose3d = coords.reshape(1, n_frames, len(unique_kpts), 3) # (n_subjects, time, bodyparts, 3)

        pose3d_dict = {'pose': pose3d, 'keypoints': unique_kpts}

        return pose3d_dict

    def generate_metadata(self):
        
        subjects = io.get_dirs(self.dataset_path)
        rows = []

        for subject in subjects: 

            if subject == 'Calibration': 
                continue 

            subject_path = os.path.join(self.dataset_path, subject)
            metadata_rows = self._get_trials(subject_path, subject)
            rows.extend(metadata_rows)

        os.makedirs('metadata', exist_ok = True)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join('metadata', f'metadata_{self.dataset_name}.csv'), index = False)

        self.metadata = df

        return df

    def select_splits(self):
        # TODO
        pass 

    def generate_dataset(self, splits = None): 

        # determine which dataset splits to generate
        valid_splits = np.unique(self.metadata['split'])

        if splits is not None: 
            splits = set(splits)
            assert splits.issubset(valid_splits) 
        else: 
            splits = valid_splits

        # generate the dataset for each split
        for split in splits: 

            subjects = io.get_dirs(self.dataset_path)

            for subject in subjects: 

                if 'Fly' not in subject: 
                    continue

                subject_path = os.path.join(self.dataset_path, subject)
                outpath = os.path.join(self.dataset_outpath, split, subject)
                os.makedirs(outpath, exist_ok = True)
                self._process_subject(subject_path, outpath, split)

                # clean up any empty directories
                if len(os.listdir(outpath)) == 0:
                    os.rmdir(outpath)

    def get_metadata(self):
        return self.metadata
    
    def set_metadata(self, df): 
        self.metadata = df 

    def _get_trials(self, subject_path, subject): 

        intrinsics_dict, *_ = self.load_calibration(self.dataset_path)
        n_cams = len(intrinsics_dict)

        video_paths = sorted(glob.glob(os.path.join(subject_path, 'videos-raw-compressed', '*.mp4')))
        unique_trials = set()
        rows = []

        for i, video_path in enumerate(video_paths):

            trial = os.path.splitext(os.path.basename(video_path))[0]
            cs = trial.split(' ')
            trial = f'{cs[0]} {cs[1]}  {cs[3]} {cs[4]}'

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

        # select subset of metadata associated with the split 
        metadata = self.metadata[self.metadata['split'] == split]

        # load calibration data
        intrinsics, extrinsics, distortions, offset_dict = self.load_calibration(self.dataset_path)
        cam_names = list(intrinsics.keys())
        video_info_dict = defaultdict(dict)

        video_paths = sorted(glob.glob(os.path.join(subject_path, 'videos-raw-compressed', f'*.mp4')))
        trials = set()

        for i, video_path in enumerate(video_paths): 

            trial = os.path.splitext(os.path.basename(video_path))[0]
            cs = trial.split(' ')
            trial = f'{cs[0]} {cs[1]}  {cs[3]} {cs[4]}'
            trials.add(trial)

        # TODO: handle dataset split: trial should ideally be moved to outer loop
        # traverse the camera names
        for trial in trials: 

            # get videos from each camera corresponding to this trial
            cs = os.path.basename(trial).split(' ')
            cam_videos = os.path.join(subject_path, 'videos-raw-compressed', f'{cs[0]} {cs[1]}*{cs[3]} {cs[4]}.mp4')
            cam_videos = sorted(glob.glob(cam_videos))

            # skip trial if metadata excludes it 
            df = metadata[metadata['id'] == trial]
            if df.empty or not df['include'].values[0]: 
                # print('skipping...')
                continue

            # load and format the 3d annotations
            trial_outpath = os.path.join(outpath, trial)
            os.makedirs(trial_outpath, exist_ok = True)
            data_path = os.path.join(subject_path, 'pose-3d', f'{trial}.csv')
            pose_dict = self.load_pose3d(data_path)
            io.save_npz(pose_dict, trial_outpath, fname = 'pose3d')

            if split == 'test':  
                # for test set, save as videos
                video_info = self._process_subject_test(
                    cam_videos, trial_outpath)
            else: 
                # for train and validation sets, deserialize the camera videos 
                # and save as images  
                video_info = self._process_subject_train(
                    cam_videos, trial_outpath)

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
        

    def _process_subject_train(self, video_paths, trial_outpath): 

        cam_height_dict = {}
        cam_width_dict = {}
        num_frames = []
        fps = []

        for cam_video_path in video_paths: 
            
            # extract info from the video   
            cam_trial = os.path.splitext(os.path.basename(cam_video_path))[0] 
            cam_name = cam_trial.split(' ')[2].split('-')[1]

            # deserialize video into images
            cam_outpath = os.path.join(trial_outpath, 'img', cam_name)
            os.makedirs(cam_outpath, exist_ok = True)

            video_info = io.deserialize_video(
                cam_video_path, 
                cam_outpath, 
                debug_ix = self.debug_ix)
            
            cam_height_dict[cam_name] = video_info['camera_height']
            cam_width_dict[cam_name] = video_info['camera_width']
            num_frames.append(video_info['num_frames'])
            fps.append(video_info['fps'])

        video_info = {
            'cam_height_dict': cam_height_dict, 
            'cam_width_dict': cam_width_dict, 
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
        os.path.makedirs(outpath, exist_ok = True)

        for cam_video_path in video_paths: 

            # extract info from the video   
            cam_trial = os.path.splitext(os.path.basename(cam_video_path))[0] 
            cam_name = cam_trial.split(' ')[2].split('-')[1]

            video_info = io.get_video_info(cam_video_path)
            cam_height_dict[cam_name] = video_info['camera_height']
            cam_width_dict[cam_name] = video_info['camera_width']
            num_frames.append(video_info['num_frames'])
            fps.append(video_info['fps'])

            # copy video to desired location
            cam_video_outpath = os.path.join(outpath, f'{cam_name}.mp4')
            shutil.copy2(cam_video_path, cam_video_outpath)

        video_info = {
            'cam_height_dict': cam_height_dict, 
            'cam_width_dict': cam_width_dict, 
            'num_frames': min(num_frames),
            'fps': min(fps)
        }

        return video_info