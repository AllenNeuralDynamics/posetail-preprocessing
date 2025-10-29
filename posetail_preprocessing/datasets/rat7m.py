import glob
import os 
import cv2
import scipy

import numpy as np
import pandas as pd 

from posetail_preprocessing.datasets import BaseDataset
from posetail_preprocessing.utils import io, assemble_extrinsics


class Rat7MDataset(BaseDataset): 

    def __init__(self, dataset_path, dataset_outpath, 
                 dataset_name = 'rat7m', debug_ix = None):
        super().__init__(dataset_path, dataset_outpath)

        self.dataset_name = dataset_name
        self.metadata = None
        self.debug_ix = debug_ix
    
    def load_calibration(self, calib_path):

        mat = scipy.io.loadmat(calib_path)

        intrinsics_dict = {}
        extrinsics_dict = {}
        distortions_dict = {}

        names = [x.lower() for x in mat['cameras'].dtype.names]
        cam_order = [0, 1, 4, 2, 3, 5]

        for i, cam in enumerate(cam_order):
            
            cam_name = names[i]
            params = mat['cameras'].item()[cam]

            intrinsics = params['IntrinsicMatrix'].item().transpose()
            rotation_matrix = params['rotationMatrix'].item().transpose()
            tvec = params['translationVector'].item()

            extrinsics = assemble_extrinsics(rotation_matrix, tvec)
            rvec = cv2.Rodrigues(rotation_matrix)[0].T[0]

            distortions = np.array([
                params['RadialDistortion'].item()[0,0],
                params['RadialDistortion'].item()[0,1],
                params['TangentialDistortion'].item()[0,0],
                params['TangentialDistortion'].item()[0,1], 0.0
            ])

            intrinsics_dict[cam_name] = intrinsics.tolist()
            extrinsics_dict[cam_name] = extrinsics.tolist()
            distortions_dict[cam_name] = distortions.tolist()

        return intrinsics_dict, extrinsics_dict, distortions_dict


    def load_pose3d(self, data_path):

        mat = scipy.io.loadmat(data_path)
        d = dict(zip(mat['mocap'].dtype.names, mat['mocap'].item()))
        bodyparts = mat['mocap'].dtype.names

        coords = []
        for bp in bodyparts:
            coords.append(d[bp])

        coords = np.array(coords)
        pose3d = np.expand_dims(coords.swapaxes(0, 1), axis = 0) # (n_subjects, time, bodyparts, 3)

        pose3d_dict = {'pose': pose3d, 'keypoints': bodyparts}

        return pose3d_dict


    def generate_metadata(self):

        video_path = os.path.join(self.dataset_path, 'videos')
        data_path = os.path.join(self.dataset_path, 'data')

        sessions = io.get_dirs(video_path)
        rows = []

        for session in sessions: 
            session_path = os.path.join(video_path, session)
            metadata_rows = self._get_videos(data_path, session_path, session)
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

    def get_metadata(self):
        return self.metadata
    
    def set_metadata(self, df): 
        self.metadata = df 

    def generate_dataset(self): 

        os.makedirs(self.dataset_outpath, exist_ok = True)
        video_path = os.path.join(self.dataset_path, 'videos')
        data_path = os.path.join(self.dataset_path, 'data')

        sessions = io.get_dirs(video_path)

        for session in sessions: 

            session_path = os.path.join(video_path, session)
            outpath = os.path.join(self.dataset_outpath, session)
            os.makedirs(outpath, exist_ok = True)
            self._process_session(data_path, session_path, outpath, session)

    
    def _get_videos(self, data_path, session_path, session): 

        rows = []

        calib_path = os.path.join(data_path, f'mocap-{session}.mat')
        intrinsics_dict, *_ = self.load_calibration(calib_path)
        n_cams = len(intrinsics_dict)

        video_paths = sorted(glob.glob(os.path.join(session_path, f'{session}-camera1-*.mp4')))

        for video_path in video_paths:

            cap = cv2.VideoCapture(video_path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            start_frame = int(os.path.splitext(os.path.basename(video_path))[0].split('-')[3])

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
    
    def _process_session(self, calib_path, session_path, outpath, session): 

        # load calibration data
        calib_path = os.path.join(calib_path, f'mocap-{session}.mat')
        intrinsics, extrinsics, distortions = self.load_calibration(calib_path)

        video_paths = sorted(glob.glob(os.path.join(session_path, f'{session}-*.mp4')))

        cam_names = np.unique([os.path.splitext(os.path.basename(video_path))[0].split('-')[2]
                               for video_path in video_paths])
        
        start_frames = np.unique([int(os.path.splitext(os.path.basename(video_path))[0].split('-')[3])
                    for video_path in video_paths])
        
        # load and format the 3d annotations
        pose_dict = self.load_pose3d(calib_path)
        pose = pose_dict['pose']
        print(session_path)

        # traverse the trials
        for start_frame in start_frames: 

            # get starting frame
            trial_outpath = os.path.join(outpath, str(start_frame).zfill(6))

            # skip video if metadata excludes it 
            if self.metadata is not None: 
                df = self.metadata[self.metadata['id'] == f'{session}_{start_frame}']
                if not df['include'].values[0]: 
                    # print('skipping...')
                    continue

            # skip if there aren't many frames remaining
            # NOTE: could also use a threshold
            if start_frame >= pose.shape[1]: 
                continue

            # load and format the 3d annotations
            pose_dict_subset = {'pose': pose[:, start_frame: start_frame + 3500, :, :], 
                                'keypoints': pose_dict['keypoints']}
            io.save_npz(pose_dict_subset, trial_outpath, fname = 'pose3d')

            # deserialize the camera videos and save as images 
            cam_height_dict = {}
            cam_width_dict = {}
            n_frames = []

            for cam_name in cam_names: 

                cam_video_path = os.path.join(session_path, f'{session}-{cam_name}-{start_frame}.mp4')
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