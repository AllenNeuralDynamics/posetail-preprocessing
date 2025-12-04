import glob
import os 
import pickle
import cv2

import numpy as np
import pandas as pd 

from posetail_preprocessing.datasets import BaseDataset
from posetail_preprocessing.utils import io, assemble_extrinsics


class POPDataset(BaseDataset): 

    def __init__(self, dataset_path, dataset_outpath, dataset_name = '3dpop', 
                 debug_ix = None):
        super().__init__(dataset_path, dataset_outpath)

        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.metadata = None
        self.debug_ix = debug_ix
    

    def load_calibration(self, calib_path):

        trial = os.path.basename(os.path.dirname(calib_path))

        intrinsics_paths = sorted(glob.glob(os.path.join(calib_path, f'{trial}*-Intrinsics.p')))
        extrinsics_paths = sorted(glob.glob(os.path.join(calib_path, f'{trial}*-Extrinsics.p')))
        # sync_paths = sorted(glob.glob(os.path.join(calib_path, f'{trial}*-SyncArray.p')))

        intrinsics_dict = {}
        extrinsics_dict = {}
        distortions_dict = {}

        for intrinsics_path, extrinsics_path in zip(intrinsics_paths, extrinsics_paths):

            cam_name = intrinsics_path.split('-')[1]
            extrinsics_cam_name = extrinsics_path.split('-')[1]

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

            subject_df = df[df.columns[df.columns.str.startswith(id)]]
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

    def select_train_set(self):
        pass 

    def select_test_set(self):  
        pass 

    def generate_dataset(self): 

        os.makedirs(self.dataset_outpath, exist_ok = True)
        subject_counts = io.get_dirs(self.dataset_path)
        rows = []

        for subject_count in subject_counts:

            subject_path = os.path.join(self.dataset_path, subject_count)
            sessions = io.get_dirs(subject_path)

            for session in sessions:  

                session_path = os.path.join(subject_path, session)
                outpath = self.dataset_outpath
                os.makedirs(outpath, exist_ok = True)
                self._process_session(session_path, outpath, session,
                                metadata = self.metadata, debug_ix = self.debug_ix)

    def get_metadata(self):
        return self.metadata
    
    def set_metadata(self, df): 
        self.metadata = df 

    def _get_splits(self, session_path, session): 

        rows = []
        n_subjects = int(session.split('_')[1].lstrip('n'))

        calib_path = os.path.join(session_path, 'CalibrationInfo')
        intrinsics_dict, *_ = self.load_calibration(calib_path)
        n_cams = len(intrinsics_dict)

        splits = io.get_dirs(os.path.join(session_path, 'TrainingSplit'))

        for split in splits: 

            split_path = os.path.join(session_path, 'TrainingSplit', split)
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
                    'split': split,
                    'include': True}
        
            rows.append(metadata_dict)

        return rows

    def _process_session(self, session_path, outpath, session,
                            metadata = None, debug_ix = None): 

        # load calibration data
        calib_path = os.path.join(session_path, 'CalibrationInfo')
        intrinsics, extrinsics, distortions = self.load_calibration(calib_path)
        cam_names = list(intrinsics.keys())

        # get splits 
        splits = io.get_dirs(os.path.join(session_path, 'TrainingSplit'))

        for split in splits: 

            split_path = os.path.join(session_path, 'TrainingSplit', split)
            video_paths = sorted(glob.glob(os.path.join(split_path, '*.mp4')))

            # check if the 3d annotations exist 
            data_path = glob.glob(os.path.join(split_path, '*Keypoint3D.csv'))[0]
            if not os.path.isfile(data_path): 
                print(f'skipping... could not find {data_path}')
                return
        
            # load and format the 3d annotations
            pose_dict = self.load_pose3d(data_path)

            # reconstruct the id
            id = f'{session}_{split}'
            trial_outpath = os.path.join(outpath, id)

            # skip video if metadata excludes it 
            if metadata is not None: 
                df = metadata[metadata['id'] == id]
                if not df['include'].values[0]: 
                    # print('skipping...')
                    continue

            io.save_npz(pose_dict, trial_outpath, fname = 'pose3d')

            # deserialize the camera videos and save as images 
            cam_height_dict = {}
            cam_width_dict = {}
            n_frames = []

            for cam_name in cam_names: 

                cam_video_path = os.path.join(split_path, f'{session}-{cam_name}.mp4')
                cam_outpath = os.path.join(trial_outpath, 'img', cam_name)

                video_info = io.deserialize_video(
                    cam_video_path, 
                    cam_outpath, 
                    start_frame = 0, 
                    debug_ix = debug_ix)

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
            io.save_yaml(data = calib_dict, outpath = trial_outpath, 
                    fname = 'metadata.yaml')

