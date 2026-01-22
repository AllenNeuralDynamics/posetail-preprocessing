import glob
import os 
import cv2
import shutil

import numpy as np
import pandas as pd 

from einops import rearrange, reduce
from posetail_preprocessing.datasets import BaseDataset
from posetail_preprocessing.utils import io, disassemble_extrinsics


class KubricMultiviewDataset(BaseDataset): 

    def __init__(self, dataset_path, dataset_outpath, 
                 dataset_name = 'kubric_multiview'):
        
        super().__init__(dataset_path, dataset_outpath)

        self.dataset_name = dataset_name
        self.metadata = None


    def load_calibration(self, calib_path):

        calib_files = sorted(glob.glob(os.path.join(calib_path, '**', 'metadata.json'), recursive = True))

        intrinsics_dict = {}
        extrinsics_dict = {}
        distortions_dict = {}
        width_dict = {}
        height_dict = {}

        for calib_file in calib_files:

            metadata = io.load_json(calib_file)
            cam_name = os.path.basename(os.path.dirname(calib_file))

            w, h = metadata['metadata']['resolution']
            intrin = np.array(metadata['camera']['K'])
            extrin = np.array(metadata['camera']['R'])
            # rvec, tvec = disassemble_extrinsics(extrin)

            intrinsics = np.diag([w, h, 1]) @ intrin @ np.diag([1, -1, -1])
            extrinsics = np.diag([1, -1, -1, 1]) @ np.linalg.inv(extrin)

            intrinsics_dict[cam_name] = intrinsics.tolist()
            extrinsics_dict[cam_name] = extrinsics.tolist()
            distortions_dict[cam_name] = np.zeros(5).tolist()
            width_dict[cam_name] = w
            height_dict[cam_name] = h
        
        return intrinsics_dict, extrinsics_dict, distortions_dict, width_dict, height_dict


    def load_pose3d(self, data_path, eps = 1e-6):

        # filter coordinates based on movement threshold
        coords = np.array(np.load(data_path)['tracks_3d'])
        total_movement = reduce(np.abs(np.diff(coords, axis = 0)), 'n k r -> k', 'sum')
        movement_check = total_movement > eps
        coords = coords[:, movement_check]
        pose3d = np.expand_dims(coords, axis = 0) # (n_subjects, time, keypoints, 3)

        keypoints = [f'kpt{i}' for i in range(pose3d.shape[2])]
        pose3d_dict = {'pose': pose3d, 'keypoints': keypoints}

        return pose3d_dict 


    def generate_metadata(self):
        
        rows = []
        splits = io.get_dirs(self.dataset_path)

        for split in splits:

            sessions = io.get_dirs(os.path.join(self.dataset_path, split))

            for i, session in enumerate(sessions): 

                print(f'{i}/{len(sessions)}')

                # make sure data has been generated for this session
                cams = io.get_dirs(os.path.join(self.dataset_path, split, session))
                if len(cams) == 0: 
                    continue

                session_path = os.path.join(self.dataset_path, split, session)
                metadata_rows = self._get_session(session_path, session, split)
                rows.extend(metadata_rows)

        os.makedirs('metadata', exist_ok = True)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join('metadata', f'metadata_{self.dataset_name}.csv'), index = False)

        self.metadata = df

        return df

    def select_splits(self):  
        # TODO 
        # mostly handled in self._get_session(), subsample here as necessary
        return self.metadata 

    def select_train_set(self, n_train_videos = 25, seed = 3):
        '''     
        NOTE: this is for testing the network on a small, specific 
        subset of the data and will likely be deprecated
        at some point

        randomly samples n videos to be curated as the training
        set. the remaining samples become validation.
        '''

        np.random.seed(seed)

        # determine train set
        train_ixs = np.random.choice(self.metadata.index, n_train_videos, replace = False)
        train_split = self.metadata.index.isin(train_ixs)

        self.metadata.loc[train_split, 'split'] = 'train'
        self.metadata.loc[train_split, 'include'] = True

        # determine val set
        self.metadata.loc[~train_split, 'split'] = 'val'
        self.metadata.loc[~train_split, 'include'] = True

        return self.metadata

    def generate_dataset(self, splits = None): 

        # determine which dataset splits to generate
        valid_splits = np.unique(self.metadata['split'])

        if splits is not None: 
            splits = set(splits)
            assert splits.issubset(valid_splits) 
        else: 
            splits = valid_splits

        for split in splits: 

            sessions = io.get_dirs(os.path.join(self.dataset_path, split))

            for i, session in enumerate(sessions): 

                # make sure data has been generated for this session
                cams = io.get_dirs(os.path.join(self.dataset_path, split, session))
                if len(cams) == 0: 
                    continue

                session_path = os.path.join(self.dataset_path, split, session)
                outpath = os.path.join(self.dataset_outpath, split, session)
                trial_outpath = os.path.join(outpath, 'trial')
                os.makedirs(outpath, exist_ok = True)
                self._process_session(session_path, trial_outpath, session, split)

                # clean up any empty directories
                if len(os.listdir(outpath)) == 0:
                    # print(f'removing: {outpath}')
                    os.rmdir(outpath) 


    def get_metadata(self):
        return self.metadata 
    
    def set_metadata(self, df): 
        self.metadata = df 

    def _get_session(self, session_path, session, split):

        intrinsics_dict, *_ = self.load_calibration(session_path)
        cam_names = list(intrinsics_dict.keys())
        n_cams = len(cam_names)

        img_path = os.path.join(session_path, cam_names[0], 'rgba*.png')
        n_frames = len(glob.glob(img_path))

        rows = [{'id': f'{session}',
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

        # select the metadata for the given split
        metadata = self.metadata[self.metadata['split'] == split]

        # load calibration data
        intrinsics, extrinsics, distortions, widths, heights = self.load_calibration(session_path)
        cam_names = list(intrinsics.keys())

        # specify conditions to process the session and 
        # skip if metadata excludes it 
        process = True
        df = metadata[metadata['id'] == session]
        if df.empty or not df['include'].values[0]:
            print('skipping', session) 
            process = False

        if process:
            
            # load and format the 3d annotations
            data_path = os.path.join(session_path, 'tracks_3d.npz')
            pose_dict = self.load_pose3d(data_path)
            io.save_npz(pose_dict, outpath, fname = 'pose3d')

            # copy image folders to new outpath
            n_frames = []

            for cam_name in cam_names:

                img_path = os.path.join(session_path, cam_name)
                img_paths = sorted(glob.glob(os.path.join(img_path, 'rgba*.png')))
                img_outpath = os.path.join(outpath, 'img', cam_name)
                os.makedirs(img_outpath, exist_ok = True)
                n_frames.append(len(img_paths))

                for i, img in enumerate(img_paths):
                    new_img_path = os.path.join(img_outpath, f'img{str(i).zfill(6)}.png')
                    shutil.copy2(img, new_img_path) 

            cam_dict = {
                'intrinsic_matrices': intrinsics, 
                'extrinsic_matrices': extrinsics, 
                'distortion_matrices': distortions,
                'camera_heights': heights,
                'camera_widths': widths,
                'n_frames': min(n_frames), 
                'num_cameras': len(cam_names)}

            # save camera metadata
            io.save_yaml(data = cam_dict, outpath = outpath, 
                    fname = 'metadata.yaml')