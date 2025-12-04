import glob
import os 
import cv2
import toml

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

        with open(calib_path, 'r') as f:
            data = toml.load(f)

        cams = list(data.keys())

        for cam in cams: 

            if cam == 'metadata': 
                continue

            cam_data = data[cam]
            cam_name = cam_data['name']

            rvec = np.array(cam_data['rotation'])
            tvec = np.array(cam_data['translation'])

            rotation_matrix, _ = cv2.Rodrigues(rvec)
            extrinsics = assemble_extrinsics(rotation_matrix, tvec)

            intrinsics_dict[cam_name] = cam_data['matrix']
            extrinsics_dict[cam_name] = extrinsics.tolist()
            distortions_dict[cam_name] = cam_data['distortions']

        return intrinsics_dict, extrinsics_dict, distortions_dict

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

        calib_path = os.path.join(self.dataset_path, 'Calibration', 'calibration.toml')
        subjects = io.get_dirs(self.dataset_path)
        rows = []

        for subject in subjects:

            if subject == 'Calibration': 
                continue 

            subject_path = os.path.join(self.dataset_path, subject)
            metadata_rows = self._get_trials(calib_path, subject_path, subject)
            rows.extend(metadata_rows)

        os.makedirs('metadata', exist_ok = True)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join('metadata', f'metadata_{self.dataset_name}.csv'), index = False)

        self.metadata = df

        return df

    def select_train_set(self):
        # TODO
        pass 

    def select_test_set(self):  
        # TODO
        pass 

    def generate_dataset(self): 

        os.makedirs(self.dataset_outpath, exist_ok = True)
        calib_path = os.path.join(self.dataset_path, 'Calibration', 'calibration.toml')
        subjects = io.get_dirs(self.dataset_path)

        for subject in subjects: 

            if 'Fly' not in subject: 
                continue

            subject_path = os.path.join(self.dataset_path, subject)
            outpath = os.path.join(self.dataset_outpath, subject)
            os.makedirs(outpath, exist_ok = True)
            self._process_subject(calib_path, subject_path, outpath)

    def get_metadata(self):
        return self.metadata
    
    def set_metadata(self, df): 
        self.metadata = df 

    def _get_trials(self, calib_path, subject_path, subject): 

        intrinsics_dict, *_ = self.load_calibration(calib_path)
        n_cams = len(intrinsics_dict)

        video_paths = sorted(glob.glob(os.path.join(subject_path, 'videos-raw-compressed', '*.mp4')))
        unique_trials = set()
        rows = []

        for i, video_path in enumerate(video_paths):

            trial = os.path.splitext(os.path.basename(video_path))[0]
            trial = ' '.join(trial.split(' ')[:2]) + '  ' + ' '.join(trial.split(' ')[3:])

            cap = cv2.VideoCapture(video_path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if trial not in unique_trials:

                metadata_dict = {
                        'id': trial,
                        'session': trial, 
                        'subject': subject, 
                        'trial': i,
                        'n_cameras': n_cams, 
                        'n_frames': n_frames,
                        'total_frames': n_frames * n_cams,
                        'split': pd.NA,
                        'include': True}
            
                unique_trials.add(trial)
                rows.append(metadata_dict)

        return rows
    

    def _process_subject(self, calib_path, subject_path, outpath): 
        
        # load calibration data
        intrinsics, extrinsics, distortions = self.load_calibration(calib_path)
        cam_names = list(intrinsics.keys())

        video_info_dict = defaultdict(dict)

        # traverse the camera names
        for cam_name in cam_names: 

            video_paths = sorted(glob.glob(os.path.join(subject_path, 'videos-raw-compressed', f'*Cam-{cam_name}*.mp4')))

            # traverse the videos associated with the camera
            for video_path in video_paths:

                # extract name of trial 
                trial = os.path.splitext(os.path.basename(video_path))[0]
                trial = ' '.join(trial.split(' ')[:2]) + '  ' + ' '.join(trial.split(' ')[3:])
                trial_outpath = os.path.join(outpath, trial)
                os.makedirs(trial_outpath, exist_ok = True)

                # skip trial if metadata excludes it 
                if self.metadata is not None: 
                    df = self.metadata[self.metadata['id'] == trial]
                    if not df['include'].values[0]: 
                        # print('skipping...')
                        continue

                # load and format the 3d annotations
                data_path = os.path.join(subject_path, 'pose-3d', f'{trial}.csv')
                pose_dict = self.load_pose3d(data_path)
                io.save_npz(pose_dict, trial_outpath, fname = 'pose3d')

                # deserialize video into images
                cam_outpath = os.path.join(trial_outpath, 'img', cam_name)
                os.makedirs(cam_outpath, exist_ok = True)

                video_info = io.deserialize_video(
                    video_path, 
                    cam_outpath, 
                    debug_ix = self.debug_ix)
                
                video_info_dict[trial][cam_name] = video_info

        # save metadata
        for trial in list(video_info_dict.keys()):

            video_info = video_info_dict[trial]
            trial_outpath = os.path.join(outpath, trial)

            cam_height_dict = {}
            cam_width_dict = {}
            n_frames = []

            for cam_name in list(video_info.keys()): 
                
                cam_height_dict[cam_name] = video_info[cam_name]['camera_height']
                cam_width_dict[cam_name] = video_info[cam_name]['camera_width']
                n_frames.append(video_info[cam_name]['num_frames'])

                calib_dict = {
                    'intrinsic_matrices': intrinsics, 
                    'extrinsic_matrices': extrinsics, 
                    'distortion_matrices': distortions,
                    'camera_heights': cam_height_dict,
                    'camera_widths': cam_width_dict,
                    'num_frames': min(n_frames), 
                    'num_cameras': len(intrinsics)}

                # save camera metadata
                io.save_yaml(data = calib_dict, outpath = trial_outpath, 
                        fname = 'metadata.yaml')
            
