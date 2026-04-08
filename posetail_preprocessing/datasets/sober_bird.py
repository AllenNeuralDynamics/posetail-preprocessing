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

import re

# functions from anipose 
def true_basename(fname):
    basename = os.path.basename(fname)
    basename = os.path.splitext(basename)[0]
    return basename


class SoberBirdDataset(BaseDataset): 

    def __init__(self, dataset_path, dataset_outpath, 
                 dataset_name = 'sober-zebrafinch', error_thresh = None):
        super().__init__(dataset_path, dataset_outpath)

        self.dataset_name = dataset_name
        self.error_thresh = error_thresh

    def load_calibration(self, calib_path):

        intrinsics_dict = {}
        extrinsics_dict = {}
        distortions_dict = {}

        calib_file = os.path.join(calib_path, 'calibration.toml')

        with open(calib_file, 'r') as f:
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

        kpts = sorted([col for col in df.columns if col.endswith('_x')
                    or col.endswith('_y') or col.endswith('_z')])
        unique_kpts = np.unique([kpt.split('_')[0] for kpt in kpts])

        coords = df[kpts].values
        n_frames = coords.shape[0]
        n_kpts = len(unique_kpts)

        pose3d = rearrange(coords, 't (n r) -> 1 t n r', t = n_frames, r = 3)  # (n_subjects, time, kpts, 3)
        pose3d_dict = {'pose': pose3d, 'keypoints': unique_kpts}

        return pose3d_dict


    def generate_metadata(self):
        
        # subjects = io.get_dirs(self.dataset_path)
        os.chdir(self.dataset_path)
        subjects = glob.glob('05*')
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

        print(self.metadata['subject'].unique())
        print(self.metadata['trial'].unique())
        self.metadata.loc[self.metadata['trial'] == 'set1_115033', 'split'] = 'val'
        self.metadata.loc[self.metadata['trial'] == 'set2_123820', 'split'] = 'test'

        fps = 200
        self.trial_frames = {
            'set1_115033': (0, 24*fps),
            'set2_115140': (20*fps, 58*fps),
            'set3_115242': (0, 58*fps),
            'set4_115349': (0, 30*fps),
            'set1_123720': (0, 58*fps),
            'set2_123820': (0, 48*fps)
        }

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
        subjects = glob.glob('05*')

        # generate the dataset for each split
        for split in splits: 
            for subject in tqdm(subjects, desc = split): 
                subject_path = os.path.join(self.dataset_path, subject)
                outpath = os.path.join(self.dataset_outpath, split, subject)
                os.makedirs(outpath, exist_ok = True)
                self._process_subject(subject_path, outpath, split)

                # clean up any empty directories
                if len(os.listdir(outpath)) == 0:
                    os.rmdir(outpath)


    def _get_trials(self, subject_path, subject): 

        rows = []

        os.chdir(subject_path)
        trials = glob.glob("set*")
        
        for trial in trials:
            vids = glob.glob(os.path.join(subject_path, trial, 'videos', '*.avi'))
            n_cams = len(vids)
            
            video_path = vids[0]
            cap = cv2.VideoCapture(video_path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            print(trial)


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

            rows.append(metadata_dict)

        return rows
    

    def _process_subject(self, subject_path, outpath, split): 

        # number of images to generate from each video
        split_frames = None
        if self.split_frames_dict and split in self.split_frames_dict: 
            split_frames = self.split_frames_dict[split]

        # select subset of metadata associated with the split 
        metadata = self.metadata[self.metadata['split'] == split]


        os.chdir(subject_path)
        trials = sorted(glob.glob("set*"))

        # traverse the camera names
        for trial in tqdm(trials):
            trial_path = os.path.join(subject_path, trial)
            intrinsics, extrinsics, distortions = self.load_calibration(trial_path)
            # get videos from each camera corresponding to this trial
            # cs = os.path.basename(trial).split(' ')
            # cam_videos = os.path.join(subject_path, 'Raw Video', f'{cs[0]} {cs[1]}*{cs[3]} {cs[4]}.mp4')
            cam_videos = os.path.join(trial_path, 'videos', '*.avi')
            cam_videos = sorted(glob.glob(cam_videos))

            # skip trial if metadata excludes it 
            df = metadata[metadata['id'] == trial]
            if df.empty or not df['include'].values[0]: 
                # print('skipping...')
                continue

            # load and format the 3d annotations
            trial_outpath = os.path.join(outpath, trial)
            os.makedirs(trial_outpath, exist_ok = True)
            # print(os.path.join(subject_path, 'pose-3d', f'{trial}.csv'))
            data_path = glob.glob(os.path.join(trial_path, 'p3ds*.csv'))[0]

            start, end = self.trial_frames[trial]
            nframes = min(end - start, split_frames)
            
            # load and format the 3d annotations
            pose_dict = self.load_pose3d(data_path)
            pose_dict_subset = self._subset_pose_dict(
                dict(pose_dict), start_frame=start, n_frames = nframes)

            io.save_npz(pose_dict_subset, trial_outpath, fname = 'pose3d')

            video_info = self._process_subject_video(
                cam_videos, trial_outpath,
                start_frame = start, n_frames = nframes)

            calib_dict = {
                'intrinsic_matrices': intrinsics, 
                'extrinsic_matrices': extrinsics, 
                'distortion_matrices': distortions,
                'num_cameras': len(intrinsics)
            }
            calib_dict.update(video_info)

            # save camera metadata
            io.save_yaml(data = calib_dict, outpath = trial_outpath, 
                    fname = 'metadata.yaml')
        

    def _process_subject_video(self, video_paths, trial_outpath,
                               start_frame = 0, n_frames = None): 

        cam_height_dict = {}
        cam_width_dict = {}
        num_frames = []
        fps = []

        for cam_video_path in video_paths: 
            
            # extract info from the video   
            cam_trial = true_basename(cam_video_path)
            cam_name = cam_trial.split('_')[1]

            # deserialize video into images
            cam_outpath = os.path.join(trial_outpath, 'img', cam_name)
            os.makedirs(cam_outpath, exist_ok = True)

            video_info = io.deserialize_video(
                cam_video_path, 
                cam_outpath,
                start_at = start_frame, debug_ix = n_frames)
            
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
    
