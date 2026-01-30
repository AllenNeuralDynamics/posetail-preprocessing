import glob
import os 
import cv2
import scipy
import shutil

import numpy as np
import pandas as pd 

from einops import reduce

from posetail_preprocessing.datasets import BaseDataset
from posetail_preprocessing.utils import io


class DexYCBDataset(BaseDataset): 

    def __init__(self, dataset_path, dataset_outpath, 
                 dataset_name = 'dex_ycb'):
        super().__init__(dataset_path, dataset_outpath)

        self.dataset_name = dataset_name

    def load_calibration(self, calib_path):

        calib_files = sorted(glob.glob(os.path.join(calib_path, '**', 'intrinsics_extrinsics.npz'), recursive = True))

        intrinsics_dict = {}
        extrinsics_dict = {}
        distortions_dict = {}
        width_dict = {}
        height_dict = {}

        for calib_file in calib_files:

            metadata = np.load(calib_file)
            cam_name = os.path.basename(os.path.dirname(calib_file))

            img_path = glob.glob(os.path.join(os.path.dirname(calib_file), 'rgb', '*.png'))[0]
            h, w = cv2.imread(img_path).shape[:2]

            intrin = metadata['intrinsics'][:3, :3]
            extrin = metadata['extrinsics']

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

        sessions = io.get_dirs(self.dataset_path)
        rows = []

        for session in sessions:

            session_path = os.path.join(self.dataset_path, session)
            metadata_rows = self._get_sessions(session_path, session)
            rows.extend(metadata_rows)

        os.makedirs('metadata', exist_ok = True)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join('metadata', f'metadata_{self.dataset_name}.csv'), index = False)

        self.metadata = df 

        return df

    def select_splits(self, split_dict = None, split_frames_dict = None, 
                      random_state = 3):
        
        # NOTE: train/test splits were mostly curated in self._get_session
        self.split_frames_dict = split_frames_dict 

        self.metadata['split'] = 'test'
        self.metadata['include'] = True

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


        sessions = io.get_dirs(self.dataset_path)

        for split in splits:

            for session in sessions: 

                session_path = os.path.join(self.dataset_path, session)
                outpath = os.path.join(self.dataset_outpath, split, session, 'trial')
                os.makedirs(outpath, exist_ok = True)
                self._process_session(session_path, outpath, session, split) 

                # clean up any empty directories
                if len(os.listdir(outpath)) == 0:
                    # print(f'removing: {outpath}')
                    os.rmdir(outpath)


    def _get_sessions(self, session_path, session): 

        rows = []

        intrinsics_dict, *_ = self.load_calibration(session_path)
        cam_names = list(intrinsics_dict.keys())
        n_cams = len(cam_names)

        video_paths = sorted(glob.glob(os.path.join(session_path, cam_names[0], 'rgb', '*.png')))
        n_frames = len(video_paths)
        subject = session.split('-', 1)[1].split('__')[0]

        metadata_dict = {
                'id': f'{session}',
                'session': session, 
                'subject': subject, 
                'n_subjects': 1,
                'trial': 1,
                'n_cameras': n_cams, 
                'n_frames': n_frames,
                'total_frames': n_frames * n_cams,
                'split': 'train',
                'include': True}
        
        rows.append(metadata_dict)

        return rows
    
    def _process_session(self, session_path, outpath, session, split): 

        # number of images to generate from each video
        split_frames = None
        if self.split_frames_dict and split in self.split_frames_dict: 
            split_frames = self.split_frames_dict[split]

        # select the metadata for the given split
        metadata = self.metadata[self.metadata['split'] == split]

        # load calibration data
        intrinsics, extrinsics, distortions, widths, heights = self.load_calibration(session_path)
        cam_names = list(intrinsics.keys())

        # specify conditions to process the session and 
        #  skip if metadata excludes it 
        process = True 
        if metadata is not None: 
            df = metadata[metadata['id'] == session]
            if df.empty or not df['include'].values[0]:
                print('skipping', session) 
                process = False

        if process:

            # load and format the 3d annotations
            data_path = os.path.join(session_path, 'tracks_3d.npz')
            pose_dict = self.load_pose3d(data_path)
            pose_dict = self._subset_pose_dict(pose_dict, n_frames = split_frames)
            io.save_npz(pose_dict, outpath, fname = 'pose3d')

            # copy image folders to new outpath
            n_frames = []

            for cam_name in cam_names:

                img_path = os.path.join(session_path, cam_name, 'rgb')
                img_paths = sorted(glob.glob(os.path.join(img_path, '*.png')))
                img_outpath = os.path.join(outpath, 'img', cam_name)
                os.makedirs(img_outpath, exist_ok = True)
                n_frames.append(len(img_paths))

                for i, img in enumerate(img_paths):

                    if split_frames and i == split_frames: 
                        break
                    
                    new_img_path = os.path.join(img_outpath, f'img{str(i).zfill(6)}.png')
                    os.symlink(img, new_img_path) 

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
    
