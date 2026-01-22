import glob
import os 
import cv2
import shutil

import numpy as np
import pandas as pd 

from posetail_preprocessing.datasets import BaseDataset
from posetail_preprocessing.utils import io, assemble_extrinsics

class AcinosetDataset(BaseDataset): 

    def __init__(self, dataset_path, dataset_outpath,
                 dataset_name = 'acinoset', keypoints_path = None, 
                 debug_ix = None):
        super().__init__(dataset_path, dataset_outpath)

        self.dataset_name = dataset_name
        self.keypoints_path = keypoints_path
        self.metadata = None
        self.debug_ix = debug_ix
    

    def load_calibration(self, calib_path, cam_names = None):
        ''' 
        NOTE: calib_data also contains 'camera_resolution'
        '''
        # mapping from the number of cameras in the given setup 
        # to the names of the associated cameras (varies in this
        # dataset)

        calib_data = io.load_json(calib_path)
        camera_params = calib_data['cameras']

        if cam_names is None:
            cam_names = [f'cam{i}' for i in range(len(camera_params))]

        intrinsics_dict = {}
        extrinsics_dict = {}
        distortions_dict = {}

        for i, cam_name in enumerate(cam_names):

            params = camera_params[i]

            intrinsics = np.array(params['k']).transpose()
            rotation_matrix = np.array(params['r']).transpose()
            tvec = np.array(params['t'])

            extrinsics = assemble_extrinsics(rotation_matrix, tvec)
            rvec = cv2.Rodrigues(rotation_matrix)[0].T[0]

            distortions = np.array([
                params['d'][0][0],
                params['d'][1][0],
                params['d'][2][0],
                params['d'][3][0], 
                0.0])

            intrinsics_dict[cam_name] = intrinsics.tolist()
            extrinsics_dict[cam_name] = extrinsics.tolist()
            distortions_dict[cam_name] = distortions.tolist()

        return intrinsics_dict, extrinsics_dict, distortions_dict


    def load_pose3d(self, data_path):

        # load in the 3d coords
        data = pd.read_pickle(data_path)
        coords = []

        for pos in data['positions']: 
            coords.append(pos)

        pose3d = np.expand_dims(coords, axis = 0) # (n_subjects, time, keypoints, 3)

        # load the keypoints if provided 
        if self.keypoints_path is not None:
            keypoints = io.load_yaml(self.keypoints_path)
        else:
            keypoints = [f'kpt{i}' for i in range(pose3d.shape[2])]

        # combine coords and keypoints 
        pose3d_dict = {'pose': pose3d, 'keypoints': keypoints}
    
        return pose3d_dict


    def generate_metadata(self):

        sessions = io.get_dirs(self.dataset_path)
        rows = []

        for session in sessions: 

            session_path = os.path.join(self.dataset_path, session)
            dirs = io.get_dirs(session_path)

            if 'extrinsic_calib' in dirs: 
                subjects = [d for d in dirs if d != 'extrinsic_calib']

                for subject in subjects: 
                    subject_path = os.path.join(session_path, subject)
                    metadata_rows = self._get_trials(subject_path, session, subject)
                    rows.extend(metadata_rows)

            else: 
                for dir in dirs: 
                    session_sub_path = os.path.join(session_path, dir)
                    subject_dirs = io.get_dirs(session_sub_path)
                    subjects = [d for d in subject_dirs if d != 'extrinsic_calib']

                    if 'extrinsic_calib' not in subject_dirs: 
                        print(f'could not find calibration data in {session_sub_path}')

                    for subject in subjects: 
                        subject_path = os.path.join(session_sub_path, subject)
                        metadata_rows = self._get_trials(subject_path, session, subject, dir)
                        rows.extend(metadata_rows)

        os.makedirs('metadata', exist_ok = True)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join('metadata', f'metadata_{self.dataset_name}.csv'), index = False)

        self.metadata = df

        return df


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

            sessions = io.get_dirs(self.dataset_path)

            for session in sessions: 

                session_path = os.path.join(self.dataset_path, session)
                dirs = io.get_dirs(session_path)

                if 'extrinsic_calib' in dirs: 
                    subjects = [d for d in dirs if d != 'extrinsic_calib']

                    for subject in subjects: 
                        id = f'{session}__{subject}'
                        subject_path = os.path.join(session_path, subject)
                        calib_path = os.path.join(session_path, 'extrinsic_calib')
                        outpath = os.path.join(self.dataset_outpath, split, id)
                        os.makedirs(outpath, exist_ok = True)
                        self._process_subject(subject_path, calib_path, outpath, id, split)

                        # clean up any empty directories
                        if len(os.listdir(outpath)) == 0:
                            # print(outpath)
                            os.rmdir(outpath)

                else: 
                    for dir in dirs: 
                        session_sub_path = os.path.join(session_path, dir)
                        subject_dirs = io.get_dirs(session_sub_path)
                        subjects = [d for d in subject_dirs if d != 'extrinsic_calib']

                        if 'extrinsic_calib' not in subject_dirs: 
                            print(f'could not find calibration data in {session_sub_path}')

                        for subject in subjects: 
                            id = f'{session}_{dir}_{subject}'
                            subject_path = os.path.join(session_sub_path, subject)
                            calib_path = os.path.join(session_sub_path, 'extrinsic_calib')
                            outpath = os.path.join(self.dataset_outpath, split, id)
                            os.makedirs(outpath, exist_ok = True)
                            self._process_subject(subject_path, calib_path, outpath, id, split)
                    
                            # clean up any empty directories
                            if len(os.listdir(outpath)) == 0:
                                os.rmdir(outpath)


    def select_splits(self):

        subject_splits = [{'phantom', 'zorro', 'jules', 'cetane', 'lily', 'kiara',  'menya'},  
                          {'ebony'},  {'romeo', 'big_girl'}]
        splits = ['train', 'val', 'test']

        for i, subjects in enumerate(subject_splits):
            self.metadata.loc[self.metadata['subject'].isin(subjects), 'split'] = splits[i]

        return self.metadata


    def get_metadata(self):
        return self.metadata
    

    def set_metadata(self, df): 
        self.metadata = df 


    def _get_trials(self, subject_path, session, subject, orientation = ''): 

        trials = io.get_dirs(subject_path) 
        rows = []

        # traverse all trials for this subject 
        for trial in trials: 

            trial_path = os.path.join(subject_path, trial)
            cam_videos = sorted(glob.glob(os.path.join(trial_path, '*.mp4')))
            cam_names = [os.path.splitext(os.path.basename(c))[0] for c in cam_videos]
            n_cams = len(cam_names)

            # need at least 2 cams for 3d tracking
            if n_cams <= 1: 
                continue

            # check if the 3d annotations exist 
            data_path = os.path.join(trial_path, 'fte_pw', 'fte.pickle')
            if not os.path.isfile(data_path): 
                print(f'skipping... could not find {data_path}')
                continue

            cap = cv2.VideoCapture(cam_videos[0])
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            metadata_dict = {
                'id': f'{session}_{orientation}_{subject}_{trial}',
                'session': session, 
                'subject': subject, 
                'trial': trial,
                'orientation': orientation, 
                'n_cameras': n_cams, 
                'n_frames': n_frames,
                'total_frames': n_frames * n_cams,
                'split': 'train',
                'include': True}
            rows.append(metadata_dict)

        return rows
    
    def _process_subject(self, subject_path, calib_path, outpath, id, split): 

        # select subset of metadata associated with the split 
        metadata = self.metadata[self.metadata['split'] == split]

        # get all trials associated with this subject
        trials = io.get_dirs(subject_path) 

        # load calibration data
        calib_paths = sorted(glob.glob(os.path.join(calib_path, '*_sba.json')))
        calib_dict = {}

        # first get number of cams per subject then get the corresponding 
        # calib path
        for calib_path in calib_paths:
            intrinsics, extrinsics, distortions = self.load_calibration(calib_path)
            n_cams = len(list(intrinsics.keys()))
            calib_dict[n_cams] = calib_path

        # traverse the trials
        for trial in trials: 

            # skip if metadata excludes it 
            df = metadata[metadata['id'] == f'{id}_{trial}']
            if df.empty or not df['include'].values[0]: 
                # print('skipping...')
                continue

            trial_path = os.path.join(subject_path, trial)
            cam_videos = sorted(glob.glob(os.path.join(trial_path, '*.mp4')))
            cam_names = [os.path.splitext(os.path.basename(c))[0] for c in cam_videos]
            n_cams = len(cam_names)

            # need at least 2 cams for 3d tracking
            if n_cams <= 1: 
                continue

            # get the calibration parameters for the n-cam setup 
            calib_path = calib_dict[n_cams]
            intrinsics, extrinsics, distortions = self.load_calibration(calib_path, cam_names)

            # check if the 3d annotations exist 
            data_path = os.path.join(trial_path, 'fte_pw', 'fte.pickle')
            if not os.path.isfile(data_path): 
                print(f'skipping... could not find {data_path}')
                continue

            # load and format the 3d annotations
            trial_outpath = os.path.join(outpath, trial)
            pose_dict = self.load_pose3d(data_path)
            io.save_npz(pose_dict, trial_outpath, fname = 'pose3d')

            if split == 'test':  
                # for test set, save as videos
                video_info = self._process_subject_test(
                    cam_videos, trial_outpath, cam_names)
            else: 
                # for train and validation sets, deserialize the camera videos 
                # and save as images  
                video_info = self._process_subject_train(
                    cam_videos, trial_outpath, cam_names)

            cam_dict = {
                'intrinsic_matrices': intrinsics, 
                'extrinsic_matrices': extrinsics, 
                'distortion_matrices': distortions,
                'num_cameras': len(intrinsics)
            }
            cam_dict.update(video_info) # height, width, n_frames, fps

            # save camera metadata
            io.save_yaml(data = cam_dict, outpath = trial_path, 
                         fname = 'metadata.yaml')
            

    def _process_subject_train(self, cam_videos, trial_outpath, cam_names):

        # deserialize the camera videos and save as images 
        cam_height_dict = {}
        cam_width_dict = {}
        num_frames = []
        fps = []

        for cam_name, cam_video in zip(cam_names, cam_videos): 
            
            cam_outpath = os.path.join(trial_outpath, 'img', cam_name)
            video_info = io.deserialize_video(
                cam_video, 
                cam_outpath, 
                start_frame = 0, 
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
    
    def _process_subject_test(self, cam_videos, trial_outpath, cam_names): 

        cam_height_dict = {}
        cam_width_dict = {}
        num_frames = []
        fps = []

        outpath = os.path.join(trial_outpath, 'vid')
        os.makedirs(outpath, exist_ok = True)

        for cam_name, cam_video in zip(cam_names, cam_videos): 

            # extract info from the video     
            video_info = io.get_video_info(cam_video)
            cam_height_dict[cam_name] = video_info['camera_height']
            cam_width_dict[cam_name] = video_info['camera_width']
            num_frames.append(video_info['num_frames'])
            fps.append(video_info['fps'])

            # copy video to desired location
            cam_video_outpath = os.path.join(outpath, f'{cam_name}.mp4')
            shutil.copy2(cam_video, cam_video_outpath)

        video_info = {
            'cam_height_dict': cam_height_dict, 
            'cam_width_dict': cam_width_dict, 
            'num_frames': min(num_frames),
            'fps': min(fps)
        }

        return video_info