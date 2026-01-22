import glob
import os 
import cv2
import shutil

import numpy as np
import pandas as pd 

from posetail_preprocessing.datasets import BaseDataset
from posetail_preprocessing.utils import io, assemble_extrinsics

class PairR24MDataset(BaseDataset): 

    def __init__(self, dataset_path, dataset_outpath, 
                 dataset_name = 'pairr24m', debug_ix = None, 
                 keypoint_format = 'absolutePosition'):
        super().__init__(dataset_path, dataset_outpath)

        self.dataset_name = dataset_name
        self.metadata = None
        self.debug_ix = debug_ix

        self.keypoint_format = keypoint_format # absolutePosition or relativePosition
    
    def load_calibration(self, calib_path):

        calib_paths = sorted(glob.glob(os.path.join(calib_path, '*.json')))

        intrinsics_dict = {}
        extrinsics_dict = {}
        distortions_dict = {}

        for path in calib_paths: 

            params = io.load_json(path)
            cam_name = os.path.splitext(os.path.basename(path))[0].split('_')[0].capitalize()

            intrinsics = np.array(params['intrinsicMatrix']).transpose()
            rotation_matrix = np.array(params['rotationMatrix']).transpose()
            tvec = np.array(params['translationVector'])

            extrinsics = assemble_extrinsics(rotation_matrix, tvec)
            rvec = cv2.Rodrigues(rotation_matrix)[0].T[0]

            distortions = np.array([
                params['radialDistortion'][0],
                params['radialDistortion'][1],
                params['tangentialDistortion'][0],
                params['tangentialDistortion'][1], 
                0.0])

            intrinsics_dict[cam_name] = intrinsics.tolist()
            extrinsics_dict[cam_name] = extrinsics.tolist()
            distortions_dict[cam_name] = distortions.tolist()

        return intrinsics_dict, extrinsics_dict, distortions_dict

    def load_pose3d(self, data_path, fmt = 'absolutePosition'):

        df = pd.read_csv(data_path)
        columns = list(df.columns)

        keypoints = self._get_keypoints(columns, fmt = fmt)
        subjects = ['_an1_', '_an2_']
        subject_pose = []

        for subject in subjects: 
            pose3d = self._get_subject_pose(df, keypoints, subject = subject)
            subject_pose.append(pose3d)

        keypoints = [kpt.replace(fmt + subjects[0], '') for kpt in keypoints]
        keypoints = [kpt.replace(fmt + subjects[1], '') for kpt in keypoints]
        keypoints = pd.unique(np.array([kpt.rstrip('_xyz') for kpt in keypoints]))

        pose3d = np.stack(subject_pose, axis = 0) # (2, frame, kpts, 3)
        pose3d_dict = {'pose': pose3d, 'keypoints': keypoints}

        return pose3d_dict


    def generate_metadata(self):

        sessions = io.get_dirs(self.dataset_path)
        rows = []

        for session in sessions: 

            # skip this session because there are only 3 cameras and missing
            # videos for cam 3 - only session like this 
            if session == '20210205_Recording_SR1_SR6m_wvid_social_new': 
                continue

            session_path = os.path.join(self.dataset_path, session)
            metadata_rows = self._get_videos(session_path, session)
            rows.extend(metadata_rows)

        os.makedirs('metadata', exist_ok = True)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join('metadata', f'metadata_{self.dataset_name}.csv'), index = False)

        self.metadata = df

        return df

    def select_splits(self):
        
        val_mask = (self.metadata['session'].str.contains('SR9') &
                    self.metadata['session'].str.contains('SR11'))
        
        test_mask = (self.metadata['session'].str.contains('SR10') & 
                     self.metadata['session'].str.contains('SR11'))
        
        nan_mask = (self.metadata['session'].str.contains('SR9') & 
                    self.metadata['session'].str.contains('SR10'))

        self.metadata.loc[val_mask, 'split'] = 'val'
        self.metadata.loc[test_mask, 'split'] = 'test'
        self.metadata.loc[nan_mask, 'split'] = pd.NA

        return self.metadata

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

            # skips sessions we aren't using
            if pd.isna(split): 
                continue 

            os.makedirs(self.dataset_outpath, exist_ok = True)
            sessions = io.get_dirs(self.dataset_path)

            for session in sessions:

                # skip this session because there are only 3 cameras and missing
                # videos for cam 3 - only session like this 
                if session == '20210205_Recording_SR1_SR6m_wvid_social_new': 
                    continue 

                session_path = os.path.join(self.dataset_path, session)
                outpath = os.path.join(self.dataset_outpath, split, session)
                os.makedirs(outpath, exist_ok = True)
                self._process_session(session_path, outpath, session, split) 

                # clean up any empty directories
                if len(os.listdir(outpath)) == 0:
                    os.rmdir(outpath)

    def get_metadata(self):
        return self.metadata
    
    def set_metadata(self, df): 
        self.metadata = df 


    def _get_keypoints(self, keypoints, fmt = None, subject = None): 

        kpts = [kpt for kpt in keypoints if kpt.endswith('_x')
                or kpt.endswith('_y') or kpt.endswith('_z')]

        if fmt: 
            assert (fmt == 'absolutePosition') or (fmt == 'relativePosition') # TODO: confirm which one
            kpts = [kpt for kpt in kpts if fmt in kpt] 

        if subject: 
            assert ('_an' in subject)
            kpts = [kpt for kpt in kpts if subject in kpt]

        return kpts


    def _get_subject_pose(self, df, keypoints, subject): 

        subject_keypoints = self._get_keypoints(keypoints, subject = subject)
        pose_df = df[subject_keypoints]

        x_cols = pose_df.columns.str.endswith('_x')
        y_cols = pose_df.columns.str.endswith('_y')
        z_cols = pose_df.columns.str.endswith('_z')

        pose_x = np.array(pose_df.loc[:, x_cols].values)
        pose_y = np.array(pose_df.loc[:, y_cols].values)
        pose_z = np.array(pose_df.loc[:, z_cols].values)

        pose3d = np.stack((pose_x, pose_y, pose_z), axis = -1)  # (frame, kpts, 3)

        return pose3d

    def _get_videos(self, session_path, session): 

        rows = []

        calib_path = os.path.join(session_path, 'calibration')
        intrinsics_dict, *_ = self.load_calibration(calib_path)
        cam_names = list(intrinsics_dict.keys())
        n_cams = len(cam_names)

        ## check if the 3d annotations exist 
        # data_path = os.path.join(session_path, 'markerDataset.csv')
        # if not os.path.isfile(data_path): 
        #     print(f'skipping... could not find {data_path}')
        #     continue

        video_paths = sorted(glob.glob(os.path.join(session_path, 'videos', cam_names[0], '*.mp4')))

        for video_path in video_paths:

            cap = cv2.VideoCapture(video_path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            start_frame = int(os.path.splitext(os.path.basename(video_path))[0])

            metadata_dict = {
                    'id': f'{session}_{start_frame}',
                    'session': session, 
                    'subject': session, 
                    'trial': 1,
                    'n_cameras': n_cams, 
                    'n_frames': n_frames,
                    'total_frames': n_frames * n_cams,
                    'split': 'train',
                    'include': True}
        
            rows.append(metadata_dict)

        return rows
    
    def _process_session(self, session_path, outpath, id, split, 
                         chunk_size = 3500): 

        # select subset of metadata associated with the split 
        metadata = self.metadata[self.metadata['split'] == split]

        # load calibration data
        calib_path = os.path.join(session_path, 'calibration')
        intrinsics, extrinsics, distortions = self.load_calibration(calib_path)

        video_dir = os.path.join(session_path, 'videos')
        cam_names = io.get_dirs(video_dir) 
        video_paths = sorted(glob.glob(os.path.join(video_dir, cam_names[0], '*.mp4')))
        start_frames = [int(os.path.splitext(os.path.basename(video_path))[0]) for video_path in video_paths]

        # check if the 3d annotations exist 
        data_path = os.path.join(session_path, 'markerDataset.csv')
        if not os.path.isfile(data_path): 
            print(f'skipping... could not find {data_path}')
            return
        
        # load and format the 3d annotations
        pose_dict = self.load_pose3d(data_path, fmt = self.keypoint_format)
        pose = pose_dict['pose']

        # traverse the trials
        for start_frame in start_frames: 
           
            # skip video if metadata excludes it 
            df = metadata[metadata['id'] == f'{id}_{start_frame}']
            if df.empty or not df['include'].values[0]:
                # print('skipping...')
                continue

            # load and format the 3d annotations
            trial_outpath = os.path.join(outpath, str(start_frame))

            pose_dict_subset = {'pose': pose[start_frame: start_frame + chunk_size, :, :], 
                                'keypoints': pose_dict['keypoints']}
            io.save_npz(pose_dict_subset, trial_outpath, fname = 'pose3d')

            # put videos/frames in the desired format
            if split == 'test':  
                # for test set, save as videos
                video_info = self._process_session_test(
                    session_path, trial_outpath, 
                    cam_names, start_frame)
            else:
                # for train and validation sets, deserialize the camera videos 
                # and save as images  
                video_info = self._process_session_train(
                    session_path, trial_outpath, 
                    cam_names, start_frame)

            calib_dict = {
                'intrinsic_matrices': intrinsics, 
                'extrinsic_matrices': extrinsics, 
                'distortion_matrices': distortions,
                'num_cameras': len(intrinsics)}
            calib_dict.update(video_info)

            # save camera metadata
            io.save_yaml(data = calib_dict, outpath = trial_outpath, 
                         fname = 'metadata.yaml')
            

    def _process_session_train(self, video_dir, trial_outpath, 
                               cam_names, start_frame): 

        # deserialize the camera videos and save as images 
        cam_height_dict = {}
        cam_width_dict = {}
        num_frames = []
        fps = []

        for cam_name in cam_names: 

            cam_video_path = os.path.join(video_dir, 'videos', cam_name, f'{start_frame}.mp4')
            cam_outpath = os.path.join(trial_outpath, 'img', cam_name)
            os.makedirs(cam_outpath, exist_ok = True)

            video_info = io.deserialize_video(
                cam_video_path, 
                cam_outpath, 
                start_frame = 0, 
                debug_ix = self.debug_ix)

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
    

    def _process_session_test(self, video_dir, trial_outpath, 
                              cam_names, start_frame): 

        # save video/image data in the expected format
        video_outpath = os.path.join(trial_outpath, 'vid')
        os.makedirs(video_outpath, exist_ok = True)

        # deserialize the camera videos and save as images 
        cam_height_dict = {}
        cam_width_dict = {}
        num_frames = []
        fps = []

        for cam_name in cam_names: 

            cam_video_path = os.path.join(video_dir, 'videos', cam_name, f'{start_frame}.mp4')
            cam_video_outpath = os.path.join(video_outpath, f'{cam_name}.mp4')

            # extract info from the video     
            video_info = io.get_video_info(cam_video_path)
            cam_height_dict[cam_name] = video_info['camera_heights']
            cam_width_dict[cam_name] = video_info['camera_widths']
            num_frames.append(video_info['num_frames'])
            fps.append(video_info['fps'])

            # copy video to desired location
            shutil.copy2(cam_video_path, cam_video_outpath)

        video_info = {
            'cam_heights': cam_height_dict, 
            'cam_widths': cam_width_dict, 
            'num_frames': min(num_frames),
            'fps': min(fps)
        }

        return video_info