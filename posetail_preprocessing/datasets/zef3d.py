import glob
import os 
import cv2
import shutil 

import numpy as np
import pandas as pd 

from posetail_preprocessing.datasets import BaseDataset
from posetail_preprocessing.utils import io, assemble_extrinsics


class ZefDataset(BaseDataset): 

    def __init__(self, dataset_path, dataset_outpath, 
                 dataset_name = '3dzef', debug_ix = None):
        super().__init__(dataset_path, dataset_outpath)

        self.dataset_name = dataset_name
        self.metadata = None
        self.debug_ix = debug_ix
    

    def load_calibration(self, calib_path): 

        calib_files = sorted(glob.glob(os.path.join(calib_path, '*.json')))
        cam_names = np.unique([os.path.basename(f).split('_')[0] for f in calib_files]).tolist()

        intrinsics_dict = {}
        extrinsics_dict = {}
        distortions_dict = {}

        for i, cam_name in enumerate(cam_names):

            cam_calib_files = [f for f in calib_files if cam_name in f]
            
            intrinsics, extrinsics, distortions = self._solve_for_extrinsics(
                calib_path_intrinsic = cam_calib_files[0], 
                calib_path_extrinsic = cam_calib_files[1])

            intrinsics_dict[cam_name] = intrinsics.tolist()
            extrinsics_dict[cam_name] = extrinsics.tolist()
            distortions_dict[cam_name] = distortions.tolist()

        return intrinsics_dict, extrinsics_dict, distortions_dict 


    def load_pose3d(self, data_path):

        df = pd.read_csv(data_path, sep = ',', header = None)
        df = df.loc[:, 0:4]
        df = df.rename(columns = {0: 'frame', 1: 'subject', 
                                2: 'head_x', 3: 'head_y', 4: 'head_z'})

        subjects = df['subject'].unique()
        df = df.pivot(index = 'frame', columns = 'subject', values = ['head_x', 'head_y', 'head_z'])
        columns = [f'sub{subj}_{col}' for col, subj in df.columns]
        df.columns = columns
        pose_df = df[sorted(columns)]
        subject_pose = []

        for subject in subjects: 

            sub_df = pose_df[[col for col in df.columns if col.startswith(f'sub{subject}_')]]

            x_cols = sub_df.columns.str.endswith('_x')
            y_cols = sub_df.columns.str.endswith('_y')
            z_cols = sub_df.columns.str.endswith('_z')

            pose_x = np.array(sub_df.loc[:, x_cols].values)
            pose_y = np.array(sub_df.loc[:, y_cols].values)
            pose_z = np.array(sub_df.loc[:, z_cols].values)

            pose3d = np.stack((pose_x, pose_y, pose_z), axis = -1)  # (frame, kpts, 3)
            subject_pose.append(pose3d)

        pose3d = np.stack(subject_pose, axis = 0) # (n_subjects, frame, kpts, 3)
        keypoints = ['head']
        pose3d_dict = {'pose': pose3d, 'keypoints': keypoints}

        return pose3d_dict 

    def generate_metadata(self):

        splits = io.get_dirs(self.dataset_path)
        rows = []

        for split in splits: 

            split_path = os.path.join(self.dataset_path, split)
            sessions = io.get_dirs(split_path)

            for session in sessions: 

                session_path = os.path.join(split_path, session)
                metadata_rows = self._get_session(session_path, session, split)
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
        splits = io.get_dirs(self.dataset_path)

        for split in splits: 

            split_path = os.path.join(self.dataset_path, split)
            sessions = io.get_dirs(split_path)

            for session in sessions: 

                session_path = os.path.join(split_path, session)
                outpath = os.path.join(self.dataset_outpath, session)
                os.makedirs(outpath, exist_ok = True)
                self._process_session(session_path, outpath, session)

    def get_metadata(self):
        return self.metadata
    
    def set_metadata(self, df): 
        self.metadata = df 

    def _solve_for_extrinsics(self, calib_path_intrinsic, calib_path_extrinsic):

        calib_data_intrinsic = io.load_json5(calib_path_intrinsic)
        calib_data_extrinsic = io.load_json5(calib_path_extrinsic)

        image_points = np.array([[c['camera']['x'], c['camera']['y']] for c in calib_data_extrinsic])
        world_points = np.array([[c['world']['x'], c['world']['y'], c['world']['z']] for c in calib_data_extrinsic])

        intrinsics = np.array(calib_data_intrinsic['K'])
        distortions = np.array(calib_data_intrinsic['Distortion']).T

        success, rvec, tvec = cv2.solvePnP(
            world_points, 
            image_points, 
            intrinsics, 
            distortions, 
            flags = cv2.SOLVEPNP_AP3P)

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        extrinsics = assemble_extrinsics(rotation_matrix, tvec)

        return intrinsics, extrinsics, distortions
    
    def _get_session(self, session_path, session, split): 

        rows = []

        intrinsics_dict, *_ = self.load_calibration(session_path)
        n_cams = len(intrinsics_dict)

        ## check if the 3d annotations exist 
        # data_path = os.path.join(session_path, 'gt', 'gt.txt')
        # if not os.path.isfile(data_path): 
        #     print(f'skipping... could not find {data_path}')
        #     continue

        img_path = os.path.join(session_path, 'imgF', '*.jpg')
        n_frames = int(len(glob.glob(img_path)))

        metadata_dict = {
                'id': f'{session}',
                'session': session, 
                'subject': session, 
                'trial': 1,
                'n_cameras': n_cams, 
                'n_frames': n_frames,
                'total_frames': n_frames * n_cams,
                'split': split, 
                'include': True}
        
        rows.append(metadata_dict)

        return rows
    

    def _process_session(self, session_path, outpath, session): 

        # specify conditions to process the session
        process = True

        # load calibration data
        intrinsics, extrinsics, distortions = self.load_calibration(session_path)

        # skip if metadata excludes it 
        if self.metadata is not None: 
            df = self.metadata[self.metadata['id'] == session]
            if not df['include'].values[0]: 
                process = False

        # check if the 3d annotations exist 
        data_path = os.path.join(session_path, 'gt', 'gt.txt')
        if not os.path.isfile(data_path): 
            print(f'skipping... could not find {data_path}')
            process = False

        if process:
            
            # load and format the 3d annotations
            pose_dict = self.load_pose3d(data_path)
            io.save_npz(pose_dict, outpath, fname = 'pose3d')

            # copy image folders to new outpath
            cam_names = ['F', 'T']
            cam_height_dict = {}
            cam_width_dict = {}
            n_frames = []

            for cam_name in cam_names:

                img_path = os.path.join(session_path, f'img{cam_name}')
                imgs = sorted(glob.glob(os.path.join(img_path, '*.jpg')))
                img = cv2.imread(imgs[0])

                cam_height_dict[cam_name] = img.shape[0]
                cam_width_dict[cam_name] = img.shape[1]
                n_frames.append(len(imgs))

                shutil.copytree(img_path, outpath, dirs_exist_ok = True)

            cam_dict = {
                'intrinsic_matrices': intrinsics, 
                'extrinsic_matrices': extrinsics, 
                'distortion_matrices': distortions,
                'camera_heights': cam_height_dict,
                'camera_widths': cam_width_dict,
                'n_frames': min(n_frames), 
                'num_cameras': len(intrinsics)}

            # save camera metadata
            io.save_yaml(data = cam_dict, outpath = outpath, 
                    fname = 'metadata.yaml')
