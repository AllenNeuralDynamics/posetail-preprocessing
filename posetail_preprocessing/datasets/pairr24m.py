import glob
import os 
import cv2

import numpy as np
import pandas as pd 

from posetail_preprocessing.datasets import BaseDataset
from posetail_preprocessing.utils import io, assemble_extrinsics

class PairR24MDataset(BaseDataset): 

    def __init__(self, dataset_path, dataset_outpath, 
                 dataset_name = 'pairr24m', debug_ix = None):
        super().__init__(dataset_path, dataset_outpath)

        self.dataset_name = dataset_name
        self.metadata = None
        self.debug_ix = debug_ix
    
    def load_calibration(self, calib_path):

        calib_paths = sorted(glob.glob(os.path.join(calib_path, '*.json')))

        intrinsics_dict = {}
        extrinsics_dict = {}
        distortions_dict = {}

        for path in calib_paths: 

            params = io.load_json(path)
            cam_name = os.path.splitext(os.path.basename(path))[0].split('_')[0]

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

    def load_pose3d(self, data_path, substring = 'absolutePosition'):

        df = pd.read_csv(data_path)
        columns = list(df.columns)

        keypoints = self._get_keypoints(columns, substring = substring)
        subjects = ['_an1_', '_an2_']
        subject_pose = []

        for subject in subjects: 
            pose3d = self._get_subject_pose(df, keypoints, subject = subject)
            subject_pose.append(pose3d)

        keypoints = [kpt.replace(substring + subjects[0], '') for kpt in keypoints]
        keypoints = [kpt.replace(substring + subjects[1], '') for kpt in keypoints]
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

    def select_train_set(self):
        # TODO
        pass 

    def select_test_set(self):  
        # TODO
        pass 


    def generate_dataset(self): 

        os.makedirs(self.dataset_outpath, exist_ok = True)
        sessions = io.get_dirs(self.dataset_path)

        for session in sessions:

            # skip this session because there are only 3 cameras and missing
            # videos for cam 3 - only session like this 
            if session == '20210205_Recording_SR1_SR6m_wvid_social_new': 
                continue 

            session_path = os.path.join(self.dataset_path, session)
            outpath = os.path.join(self.dataset_outpath, session)
            os.makedirs(outpath, exist_ok = True)
            self._process_session(session_path, outpath, session) 

    def get_metadata(self):
        return self.metadata
    
    def set_metadata(self, df): 
        self.metadata = df 


    def _get_keypoints(self, keypoints, substring = 'absolutePosition'): 

        # TODO: absolute or relative
        kpts = [kpt for kpt in keypoints if kpt.endswith('_x')
                or kpt.endswith('_y') or kpt.endswith('_z')]
        
        kpts = [kpt for kpt in kpts if substring in kpt] 

        return kpts


    def _get_subject_pose(self, df, keypoints, subject): 

        subject_keypoints = self._get_keypoints(keypoints, substring = subject)
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
        n_cams = len(intrinsics_dict)

        ## check if the 3d annotations exist 
        # data_path = os.path.join(session_path, 'markerDataset.csv')
        # if not os.path.isfile(data_path): 
        #     print(f'skipping... could not find {data_path}')
        #     continue

        video_paths = sorted(glob.glob(os.path.join(session_path, 'videos', 'Camera1', '*.mp4')))

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
                    'split': pd.NA,
                    'include': True}
        
            rows.append(metadata_dict)

        return rows
    
    def _process_session(self, session_path, outpath, id): 

        # load calibration data
        calib_path = os.path.join(session_path, 'calibration')
        intrinsics, extrinsics, distortions = self.load_calibration(calib_path)

        video_dir = os.path.join(session_path, 'videos')
        cam_names = io.get_dirs(video_dir) 
        video_paths = sorted(glob.glob(os.path.join(video_dir, cam_names[0], '*.mp4')))

        # check if the 3d annotations exist 
        data_path = os.path.join(session_path, 'markerDataset.csv')
        if not os.path.isfile(data_path): 
            print(f'skipping... could not find {data_path}')
            return
        
        # load and format the 3d annotations
        pose_dict = self.load_pose3d(data_path)
        pose = pose_dict['pose']

        # traverse the trials
        for video_path in video_paths: 

            # get starting frame
            video_name = os.path.basename(video_path)
            start_frame = int(os.path.splitext(video_name)[0])
            trial_outpath = os.path.join(outpath, str(start_frame))

            # skip video if metadata excludes it 
            if self.metadata is not None: 
                df = self.metadata[self.metadata['id'] == f'{id}_{start_frame}']
                if not df['include'].values[0]:
                    print('skipping...')
                    continue

            # load and format the 3d annotations
            pose_dict_subset = {'pose': pose[start_frame: start_frame + 3500, :, :], 
                                'keypoints': pose_dict['keypoints']}
            io.save_npz(pose_dict_subset, trial_outpath, fname = 'pose3d')

            # deserialize the camera videos and save as images 
            cam_height_dict = {}
            cam_width_dict = {}
            n_frames = []

            for cam_name in cam_names: 

                cam_video_path = os.path.join(video_dir, cam_name, video_name)
                cam_outpath = os.path.join(trial_outpath, 'img', cam_name)

                video_info = io.deserialize_video(
                    cam_video_path, 
                    cam_outpath, 
                    start_frame = 0, 
                    debug_ix = self.debug_ix)

                cam_height_dict[cam_name] = video_info['camera_height']
                cam_width_dict[cam_name] = video_info['camera_width']
                n_frames.append(video_info['num_frames'])

            calib_dict = {
                'intrinsic_matrices': intrinsics, 
                'extrinsic_matrices': extrinsics, 
                'distortion_matrices': distortions,
                'camera_heights': cam_height_dict,
                'camera_widths': cam_width_dict,
                'num_frames': min(n_frames), 
                'num_cameras': len(intrinsics)}

            # save camera metadata
            io.save_json(data = calib_dict, outpath = trial_outpath, 
                    fname = 'metadata.yaml')