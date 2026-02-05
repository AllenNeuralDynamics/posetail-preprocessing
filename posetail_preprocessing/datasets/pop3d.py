import glob
import os 
import pickle
import cv2
import shutil

import numpy as np
import pandas as pd 

from posetail_preprocessing.datasets import BaseDataset
from posetail_preprocessing.utils import io, assemble_extrinsics


class POPDataset(BaseDataset): 

    def __init__(self, dataset_path, dataset_outpath, dataset_name = '3dpop'):
        super().__init__(dataset_path, dataset_outpath)

        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
    

    def load_calibration(self, calib_path):

        trial = os.path.basename(os.path.dirname(calib_path))

        intrinsics_paths = sorted(glob.glob(os.path.join(calib_path, f'{trial}*-Intrinsics.p')))
        extrinsics_paths = sorted(glob.glob(os.path.join(calib_path, f'{trial}*-Extrinsics.p')))
        # sync_paths = sorted(glob.glob(os.path.join(calib_path, f'{trial}*-SyncArray.p')))

        intrinsics_dict = {}
        extrinsics_dict = {}
        distortions_dict = {}

        for intrinsics_path, extrinsics_path in zip(intrinsics_paths, extrinsics_paths):

            cam_name = os.path.basename(intrinsics_path).split('-')[1]
            extrinsics_cam_name = os.path.basename(extrinsics_path).split('-')[1]

            assert cam_name == extrinsics_cam_name

            with open(intrinsics_path, 'rb') as f:
                intrinsics, distortions = pickle.load(f)

            with open(extrinsics_path, 'rb') as f:
                rvec, tvec = pickle.load(f)

            rotation_matrix, _ = cv2.Rodrigues(rvec)
            extrinsics = assemble_extrinsics(rotation_matrix, tvec)

            intrinsics_dict[cam_name] = intrinsics.tolist()
            extrinsics_dict[cam_name] = extrinsics.tolist()
            distortions_dict[cam_name] = distortions.tolist()

        return intrinsics_dict, extrinsics_dict, distortions_dict


    def load_pose3d(self, data_path):

        df = pd.read_csv(data_path)

        kpts = [col for col in df.columns if col.endswith('_x')
                or col.endswith('_y') or col.endswith('_z')]

        subject_id_kpts = np.unique([kpt.rsplit('_', 1)[0] for kpt in kpts])
        unique_kpts = np.unique([kpt.split('_', 2)[2] for kpt in subject_id_kpts])
        ids = np.unique([kpt.split('_')[0] for kpt in kpts])

        subject_coords = []

        for id in ids:

            subject_df = df[df.columns[df.columns.str.startswith(f'{id}_')]]
            coords = subject_df.values
            n_frames, _ = coords.shape

            pose3d = coords.reshape(n_frames, len(unique_kpts), 3)
            subject_coords.append(pose3d)

        subject_coords = np.stack(subject_coords)
        pose3d_dict = {'pose': subject_coords, 'keypoints': unique_kpts, 'ids': ids}

        return pose3d_dict


    def generate_metadata(self):

        subject_counts = io.get_dirs(self.dataset_path)
        rows = []

        for subject_count in subject_counts:
            
            if subject_count in {'Markerless', 'N6000', 'SinglePigeon'}:
                continue

            subject_path = os.path.join(self.dataset_path, subject_count)
            sessions = io.get_dirs(subject_path)

            for session in sessions: 

                session_path = os.path.join(subject_path, session)
                metadata_rows = self._get_splits(session_path, session)
                rows.extend(metadata_rows)

        os.makedirs('metadata', exist_ok = True)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join('metadata', f'metadata_{self.dataset_name}.csv'), index = False)

        self.metadata = df 

        return df


    def select_splits(self, split_dict = None, split_frames_dict = None, 
                      random_state = 3):
        
        self.split_frames_dict = split_frames_dict

        # splits were mostly processed in self._get_splits
        if split_dict: 
            for split, n in split_dict.items():
                self._select_subset_for_split(split = split, n = n, random_state = random_state)

        return self.metadata

    def generate_dataset(self, splits = None):

        # determine which dataset splits to generate
        valid_splits = pd.unique(self.metadata['split'])

        if splits is not None: 
            splits = set(splits)
            assert splits.issubset(valid_splits) 
        else: 
            splits = valid_splits 

        for split in splits: 

            subject_counts = io.get_dirs(self.dataset_path)

            for subject_count in subject_counts:

                if subject_count in {'Markerless', 'N6000', 'SinglePigeon'}:
                    continue

                subject_path = os.path.join(self.dataset_path, subject_count)
                subject_outpath = os.path.join(self.dataset_outpath, split, subject_count)
                sessions = io.get_dirs(subject_path)

                for session in sessions:  

                    session_path = os.path.join(subject_path, session)
                    outpath = os.path.join(subject_outpath, session)
                    self._process_session(session_path, outpath, session,
                                          metadata = self.metadata, 
                                          split = split)


    def _get_splits(self, session_path, session): 

        rows = []
        n_subjects = int(session.split('_')[1].lstrip('n'))

        calib_path = os.path.join(session_path, 'CalibrationInfo')
        intrinsics_dict, *_ = self.load_calibration(calib_path)
        n_cams = len(intrinsics_dict)

        splits = io.get_dirs(os.path.join(session_path, 'TrainingSplit'))

        for split in splits: 

            split_path = os.path.join(session_path, 'TrainingSplit', split.capitalize())
            video_paths = sorted(glob.glob(os.path.join(split_path, '*.mp4')))

            cap = cv2.VideoCapture(video_paths[0])
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            metadata_dict = {
                    'id': f'{session}_{split}',
                    'session': session, 
                    'subject':'', 
                    'n_subjects': n_subjects,
                    'trial': 1,
                    'n_cameras': n_cams, 
                    'n_frames': n_frames,
                    'total_frames': n_frames * n_cams,
                    'split': split.lower(),
                    'include': True}
        
            rows.append(metadata_dict)

        return rows

    def _process_session(self, session_path, trial_outpath, session,
                            metadata = None, split = None): 

        # number of images to generate from each video
        split_frames = None
        if self.split_frames_dict and split in self.split_frames_dict: 
            split_frames = self.split_frames_dict[split]

        # select subset of metadata associated with the split 
        metadata = self.metadata[self.metadata['split'] == split]

        # load calibration data
        calib_path = os.path.join(session_path, 'CalibrationInfo')
        intrinsics, extrinsics, distortions = self.load_calibration(calib_path)
        cam_names = list(intrinsics.keys())

        # get splits 
        split_path = os.path.join(session_path, 'TrainingSplit', split.capitalize())

        # check if the 3d annotations exist 
        data_path = glob.glob(os.path.join(split_path, '*Keypoint3D.csv'))[0]
        if not os.path.isfile(data_path): 
            # print(f'skipping... could not find {data_path}')
            return
    
        # load and format the 3d annotations
        pose_dict = self.load_pose3d(data_path)
        pose_dict = self._subset_pose_dict(pose_dict, n_frames = split_frames)

        # reconstruct the id
        id = f'{session}_{split.capitalize()}'

        # skip video if metadata excludes it 
        process = True
        df = metadata[metadata['id'] == id]
        if df.empty or not df['include'].values[0]: 
            # print('skipping...')
            process = False

        if process: 

            os.makedirs(trial_outpath, exist_ok = True)
            io.save_npz(pose_dict, trial_outpath, fname = 'pose3d')

            # put videos/frames in the desired format
            if split == 'test':  
                # for test set, save as videos
                video_info = self._process_session_test(
                    split_path, trial_outpath, session, cam_names)
            else:
                # for train and validation sets, deserialize the camera videos 
                # and save as images  
                video_info = self._process_session_train(
                    split_path, trial_outpath, session, 
                    cam_names, split_frames = split_frames
                )

            calib_dict = {
                'intrinsic_matrices': intrinsics, 
                'extrinsic_matrices': extrinsics, 
                'distortion_matrices': distortions,
                'num_cameras': len(intrinsics)}
            calib_dict.update(video_info)

            # save camera metadata
            io.save_yaml(data = calib_dict, outpath = trial_outpath, 
                    fname = 'metadata.yaml')


    def _process_session_train(self, split_path, trial_outpath, 
                               session, cam_names, split_frames = None): 

        # deserialize the camera videos and save as images 
        cam_height_dict = {}
        cam_width_dict = {}
        num_frames = []
        fps = []

        for cam_name in cam_names: 

            cam_video_path = os.path.join(split_path, f'{session}-{cam_name}.mp4')
            cam_outpath = os.path.join(trial_outpath, 'img', cam_name)

            video_info = io.deserialize_video(
                cam_video_path, 
                cam_outpath, 
                start_frame = 0, 
                debug_ix = split_frames)

            cam_height_dict[cam_name] = video_info['camera_heights']
            cam_width_dict[cam_name] = video_info['camera_widths']
            num_frames.append(video_info['num_frames'])
            fps.append(video_info['fps'])

        video_info = {
            'cam_heights': cam_height_dict, 
            'cam_widths': cam_width_dict, 
            'num_frames': min(num_frames),
            'fps': min(fps)
        }

        return video_info 
    

    def _process_session_test(self, split_path, trial_outpath, 
                               session, cam_names): 

        # deserialize the camera videos and save as images 
        cam_height_dict = {}
        cam_width_dict = {}
        num_frames = []
        fps = []

        outpath = os.path.join(trial_outpath, 'vid')
        os.makedirs(outpath, exist_ok = True)

        for cam_name in cam_names: 

            cam_video_path = os.path.join(split_path, f'{session}-{cam_name}.mp4')
            cam_video_outpath = os.path.join(outpath, f'{cam_name}.mp4')

            # extract info from the video     
            video_info = io.get_video_info(cam_video_path)
            cam_height_dict[cam_name] = video_info['camera_heights']
            cam_width_dict[cam_name] = video_info['camera_widths']
            num_frames.append(video_info['num_frames'])
            fps.append(video_info['fps'])

            # copy video to desired location
            os.symlink(cam_video_path, cam_video_outpath)

        video_info = {
            'cam_heights': cam_height_dict, 
            'cam_widths': cam_width_dict, 
            'num_frames': min(num_frames),
            'fps': min(fps)
        }

        return video_info 