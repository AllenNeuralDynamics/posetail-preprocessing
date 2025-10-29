import cv2
import glob
import itertools
import os 
import shutil 

import numpy as np
import pandas as pd 

from posetail_preprocessing.datasets import BaseDataset
from posetail_preprocessing.utils import io


class CMUPanopticGSDataset(BaseDataset): 

    def __init__(self, dataset_path, dataset_outpath, 
                 dataset_name = 'cmupanoptic_3dgs', 
                 n_cameras = 6, n_combinations = 6, n_tries = 100, 
                 seed = 11, debug_ix = None):
        super().__init__(dataset_path, dataset_outpath)

        self.dataset_name = dataset_name

        self.n_cameras = n_cameras
        self.n_combinations = n_combinations
        self.n_tries = n_tries
        self.seed = seed

        self.metadata = None
        self.debug_ix = debug_ix
    
    def load_calibration(self, calib_path, n_dist_coeffs = 5):

        cam_data = io.load_json(calib_path)

        intrinsics_dict = {}
        extrinsics_dict = {}
        distortions_dict = {}
        resolution_dict = {}

        cam_names = cam_data['cam_id'][0]

        for i, cam in enumerate(cam_names):

            width = cam_data['w']
            height = cam_data['h']
            resolution_dict[cam] = (height, width)

            intrinsics = cam_data['k'][0][i]
            extrinsics = np.array(cam_data['w2c'][0][i])[:3, :]

            # assume no distortion for gaussian splatting
            distortions = np.zeros(n_dist_coeffs)

            intrinsics_dict[cam] = intrinsics
            extrinsics_dict[cam] = extrinsics.tolist()
            distortions_dict[cam] = distortions.tolist()

        return intrinsics_dict, extrinsics_dict, distortions_dict, resolution_dict


    def load_pose3d(self, data_path):

        data = np.load(data_path)
        pose3d = np.expand_dims(data['means3D'], axis = 0) # (n_subjects, frames, kpts, 3)

        keypoints = [f'kpt{i}' for i in range(pose3d.shape[1])]
        pose3d_dict = {'pose': pose3d, 'keypoints': keypoints}

        return pose3d_dict

    def generate_metadata(self):

        data_path = os.path.join(self.dataset_path, 'images')
        sessions = io.get_dirs(data_path)
        rows = []

        for session in sessions: 

            session_path = os.path.join(data_path, session)
            metadata_rows = self._get_session(session_path, session)
            rows.extend(metadata_rows)

        os.makedirs('metadata', exist_ok = True)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join('metadata', f'metadata_{self.dataset_name}.csv'), index = False)

        self.metadata = df

        return df

    def select_train_set(self):

        test_sports = ['basketball', 'boxes']

        for sport in test_sports: 
            df.loc[df['session'] == sport, 'split'] = 'test'
            df.loc[df['session'] == sport, 'include'] = False 

        return df

    def select_test_set(self):  
        # TODO
        pass 

    def generate_dataset(self): 

        os.makedirs(self.dataset_outpath, exist_ok = True)
        sessions = io.get_dirs(os.path.join(self.dataset_path, 'images'))

        for session in sessions: 

            # get all camera names
            img_path = os.path.join(self.dataset_path, 'images', session, 'ims')
            cam_names = io.get_dirs(img_path)
            cam_names = sorted([int(i) for i in cam_names])
            
            # get subsets of cameras to process
            combinations = self._get_camera_subset(img_path, cam_names, seed = self.seed)

            outpath = os.path.join(self.dataset_outpath, session)
            os.makedirs(outpath, exist_ok = True)
            self._process_session(outpath, session, combinations)

    def get_metadata(self):
        return self.metadata
    
    def set_metadata(self, df): 
        self.metadata = df 

    def _get_session(self, session_path, session): 

        calib_paths = glob.glob(os.path.join(session_path, '*.json'))
        n_cams = 0
        rows = []

        for calib_path in calib_paths: 

            intrinsics_dict, *_ = self.load_calibration(calib_path)
            cam_names = list(intrinsics_dict.keys())
            n_cams += len(cam_names)

            img_path = os.path.join(session_path, 'ims', str(cam_names[0]), '*.jpg')
            n_frames = int(len(glob.glob(img_path)))

        metadata_dict = {
                'id': f'{session}',
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
    
    def _get_camera_subset(self, img_path, cam_names, seed = None): 

        # for reproducibility across sports and runs
        # NOTE: if you want different views sampled for each
        # run, could pass in a different seed
        np.random.seed(seed)

        # get all possible combinations of camera views
        combinations = list(itertools.combinations(cam_names, self.n_cameras))
        print(f'sampling {self.n_combinations} combinations of {self.n_cameras} cameras...')

        # find a score to maximize camera view diversity
        best_score = 0
        cam_subsets = None

        for i in range(self.n_tries):

            ixs = np.random.choice(len(combinations), size = self.n_combinations, replace = False)
            subsets = [np.array(combinations[ix]) for ix in ixs]
            score = len(np.unique(np.stack(subsets)))
            
            # track best subset 
            if score >= best_score:
                best_score = score 
                cam_subsets = subsets

        return cam_subsets

    def _process_session(self, outpath, session, combinations): 

        # specify conditions to process the session
        process = True
        session_path = os.path.join(self.dataset_path, 'images', session)
        data_path = os.path.join(self.dataset_path, 'output', 'pretrained', session, 'params.npz')
        calib_paths = glob.glob(os.path.join(session_path, '*.json'))

        for i, combination in enumerate(combinations): 

            for calib_path in calib_paths: 

                # load calibration data
                combination_outpath = os.path.join(outpath, f'cam_combination_{i}')
                intrinsics, extrinsics, distortions, _ = self.load_calibration(calib_path)
                cam_names = list(intrinsics.keys())

                # skip if metadata excludes it 
                if self.metadata is not None: 
                    df = self.metadata[self.metadata['id'] == f'{session}']
                    if not df['include'].values[0]: 
                        process = False

                if process:
                
                    # load and format the 3d annotations
                    pose_dict = self.load_pose3d(data_path)
                    io.save_npz(pose_dict, combination_outpath, fname = 'pose3d')

                    # copy image folders to new outpath
                    cam_height_dict = {}
                    cam_width_dict = {}
                    n_frames = []

                    for cam_name in cam_names:

                        # only select cameras for the current combination
                        if cam_name not in combination: 
                            continue

                        img_path = os.path.join(session_path, 'ims', str(cam_name))
                        imgs = sorted(glob.glob(os.path.join(img_path, '*.jpg')))
                        img = cv2.imread(imgs[0])

                        cam_height_dict[cam_name] = img.shape[0]
                        cam_width_dict[cam_name] = img.shape[1]
                        n_frames.append(len(imgs))

                        img_outpath = os.path.join(combination_outpath, 'img', str(cam_name))
                        shutil.copytree(img_path, img_outpath, dirs_exist_ok = True)

            cam_dict = {
                'intrinsic_matrices': intrinsics, 
                'extrinsic_matrices': extrinsics, 
                'distortion_matrices': distortions,
                'camera_heights': cam_height_dict,
                'camera_widths': cam_width_dict,
                'n_frames': min(n_frames), 
                'num_cameras': len(intrinsics)}

            # save camera metadata
            io.save_json(data = cam_dict, outpath = combination_outpath, 
                    fname = 'metadata.yaml')