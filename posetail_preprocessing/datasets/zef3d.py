import glob
import os 
import cv2
import shutil 

import numpy as np
import pandas as pd 

from tqdm import tqdm

from posetail_preprocessing.datasets import BaseDataset
from posetail_preprocessing.utils import io, assemble_extrinsics


class ZefDataset(BaseDataset): 

    def __init__(self, dataset_path, dataset_outpath, 
                 dataset_name = '3dzef'):
        super().__init__(dataset_path, dataset_outpath)

        self.dataset_name = dataset_name
    

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

    def select_splits(self, split_frames_dict = None):

        self.split_frames_dict = split_frames_dict

        splits = ['train', 'test', None]
        subject_splits = [{'ZebraFish-01', 'ZebraFish-02'},
                          {'ZebraFish-03', 'ZebraFish-04'}, 
                          {'ZebraFish-05', 'ZebraFish-06', 
                           'ZebraFish-07', 'ZebraFish-08'}]

        for i, subjects in enumerate(subject_splits):
            self.metadata.loc[self.metadata['subject'].isin(subjects), 'split'] = splits[i]

        return self.metadata

    def generate_dataset(self, splits = None): 

        # determine which dataset splits to generate
        valid_splits = pd.unique(self.metadata['split'])

        if splits is not None: 
            splits = set(splits)
            assert splits.issubset(valid_splits) 
        else: 
            splits = valid_splits

        # generate the dataset for each split
        for split in splits: 

            if split is None: 
                continue

            orig_splits = io.get_dirs(self.dataset_path)

            for orig_split in tqdm(orig_splits, desc = f'{split}_outer'): 

                orig_split_path = os.path.join(self.dataset_path, orig_split)
                sessions = io.get_dirs(orig_split_path)

                for session in tqdm(sessions, desc = f'{split}_inner'): 

                    session_path = os.path.join(orig_split_path, session)
                    outpath = os.path.join(self.dataset_outpath, split, session, 'trial')
                    os.makedirs(outpath, exist_ok = True)
                    self._process_session(session_path, outpath, session, split)
                
                    # clean up any empty directories
                    if len(os.listdir(outpath)) == 0:
                        os.rmdir(outpath)


    def _solve_for_extrinsics(self, calib_path_intrinsic, calib_path_extrinsic, 
                              method = cv2.SOLVEPNP_ITERATIVE):

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
            flags = method)

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        extrinsics = assemble_extrinsics(rotation_matrix, tvec)

        return intrinsics, extrinsics, distortions
    

    def _get_session(self, session_path, session, split): 

        intrinsics_dict, *_ = self.load_calibration(session_path)
        n_cams = len(intrinsics_dict)

        ## check if the 3d annotations exist 
        # data_path = os.path.join(session_path, 'gt', 'gt.txt')
        # if not os.path.isfile(data_path): 
        #     print(f'skipping... could not find {data_path}')
        #     continue

        img_path = os.path.join(session_path, 'imgF', '*.jpg')
        n_frames = int(len(glob.glob(img_path)))

        rows = [{
                'id': f'{session}',
                'session': session, 
                'subject': session, 
                'trial': 1,
                'n_cameras': n_cams, 
                'n_frames': n_frames,
                'total_frames': n_frames * n_cams,
                'split': split, 
                'include': True}]
        
        return rows

    def _process_session(self, session_path, outpath, session, split): 

        # number of images to generate from each video
        split_frames = None
        if self.split_frames_dict and split in self.split_frames_dict: 
            split_frames = self.split_frames_dict[split]

        # select subset of metadata associated with the split 
        metadata = self.metadata[self.metadata['split'] == split]

        # load calibration data
        intrinsics, extrinsics, distortions = self.load_calibration(session_path)

        # specify conditions to process the session and 
        # skip if metadata excludes it 
        process = True
        df = metadata[metadata['id'] == session]
        if df.empty or not df['include'].values[0]: 
            process = False

        # check if the 3d annotations exist 
        data_path = os.path.join(session_path, 'gt', 'gt.txt')
        if not os.path.isfile(data_path): 
            print(f'skipping... could not find {data_path}')
            process = False

        if process:
            
            # load and format the 3d annotations
            pose_dict = self.load_pose3d(data_path)
            pose_dict = self._subset_pose_dict(pose_dict, n_frames = split_frames)
            io.save_npz(pose_dict, outpath, fname = 'pose3d')

            # copy image folders to new outpath
            cam_names = ['F', 'T']
            cam_height_dict = {}
            cam_width_dict = {}
            n_frames = []

            for cam_name in cam_names:

                cam_outpath = os.path.join(outpath, 'img', cam_name)
                os.makedirs(cam_outpath, exist_ok = True)

                img_prefix = os.path.join(session_path, f'img{cam_name}')
                img_paths = sorted(glob.glob(os.path.join(img_prefix, '*.jpg')))
                img = cv2.imread(img_paths[0])

                cam_height_dict[cam_name] = img.shape[0]
                cam_width_dict[cam_name] = img.shape[1]
                n_frames.append(len(img_paths))

                for i, img_path in enumerate(img_paths):

                    if split_frames and i == split_frames: 
                        break 

                    cam_img_outpath = os.path.join(cam_outpath, f'img{str(i).zfill(6)}.png')
                    os.symlink(img_path, cam_img_outpath)

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
            
        
