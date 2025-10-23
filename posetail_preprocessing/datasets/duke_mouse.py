import glob
import os 
import cv2
import scipy

import numpy as np
import pandas as pd 

from posetail_preprocessing.datasets import BaseDataset
from posetail_preprocessing.utils import io, assemble_extrinsics


class DukeMouseDataset(BaseDataset): 

    def __init__(self, dataset_path, dataset_outpath, 
                 dataset_name = 'duke_mouse', debug_ix = None):
        super().__init__(dataset_path, dataset_outpath)

        self.dataset_name = dataset_name
        self.metadata = None
        self.debug_ix = debug_ix


    def load_calibration(self, calib_path):

        mat = scipy.io.loadmat(calib_path)

        if len(mat['camnames'][0]) == 6: 
            cam_names = [cam[0] for cam in mat['camnames'][0]]
        else: 
            cam_names = [cam[0][0] for cam in mat['camnames']]

        intrinsics_dict = {}
        extrinsics_dict = {}
        distortions_dict = {}

        for i, cam in enumerate(cam_names):

            params = mat['params'][i][0][0]
            
            intrinsics = params['K'][0].transpose()
            rotation_matrix = params['r'][0].transpose()
            tvec = params['t'][0]
            rvec = cv2.Rodrigues(rotation_matrix)[0].T[0]

            extrinsics = assemble_extrinsics(rotation_matrix, tvec).transpose()

            distortions = np.array([
                params['RDistort'].item()[0,0],
                params['RDistort'].item()[0,1],
                params['TDistort'].item()[0,0],
                params['TDistort'].item()[0,1], 
                0.0])

            intrinsics_dict[cam] = intrinsics.tolist()
            extrinsics_dict[cam] = extrinsics.tolist()
            distortions_dict[cam] = distortions.tolist()

        return intrinsics_dict, extrinsics_dict, distortions_dict


    def load_pose3d(self, data_path):

        data = scipy.io.loadmat(data_path)
        coords = np.array(data['pred'])
        pose3d = np.expand_dims(coords.swapaxes(0, 1), axis = 0) # (n_subjects, time, bodyparts, 3)

        bodyparts = [f'kpt{i}' for i in range(pose3d.shape[1])]
        pose3d_dict = {'pose': pose3d, 'keypoints': bodyparts}

        return pose3d_dict

    def generate_metadata(self):

        subjects = io.get_dirs(self.dataset_path)
        rows = []

        for subject in subjects: 

            subject_path = os.path.join(self.dataset_path, subject)
            metadata_rows = self._get_videos(subject_path, subject)
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
        subjects = io.get_dirs(self.dataset_path)

        for subject in subjects: 

            subject_path = os.path.join(self.dataset_path, subject)
            outpath = os.path.join(self.dataset_outpath, subject)
            os.makedirs(outpath, exist_ok = True)
            self._process_subject(subject_path, outpath, subject)

    def get_metadata(self):
        return self.metadata
    
    def set_metadata(self, df): 
        self.metadata = df 


    def _get_videos(self, subject_path, subject): 

        rows = []

        calib_path = os.path.join(subject_path, 'annotations.mat')
        intrinsics_dict, *_ = self.load_calibration(calib_path)

        n_cams = len(intrinsics_dict)
        cam_names = list(intrinsics_dict.keys())

        video_paths_1 = sorted(glob.glob(os.path.join(subject_path, 'videos', cam_names[0], '*.mp4')))
        video_paths_6 = sorted(glob.glob(os.path.join(subject_path, 'videos', cam_names[-1], '*.mp4')))

        if len(video_paths_1) == len(video_paths_6):
            video_paths = video_paths_1
            n_cams_factor = 6
        else: 
            video_paths = video_paths_6
            n_cams_factor = 1

        for video_path in video_paths:

            cap = cv2.VideoCapture(video_path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            start_frame = int(os.path.splitext(os.path.basename(video_path))[0])

            metadata_dict = {
                    'id': f'{subject}_{start_frame}',
                    'session': subject, 
                    'subject': subject, 
                    'trial': 1,
                    'n_cameras': n_cams, 
                    'n_frames': n_frames,
                    'total_frames': n_frames * n_cams_factor,
                    'split': pd.NA,
                    'include': True}
        
            rows.append(metadata_dict)

        return rows
    

    def _process_subject(self, subject_path, outpath, subject): 

        # load calibration data
        calib_path = os.path.join(subject_path, 'annotations.mat')
        intrinsics, extrinsics, distortions = self.load_calibration(calib_path)

        video_dir = os.path.join(subject_path, 'videos')
        cam_names = io.get_dirs(video_dir) 
        video_paths = sorted(glob.glob(os.path.join(video_dir, cam_names[-1], '*.mp4')))
        alt_video_paths = sorted(glob.glob(os.path.join(video_dir, cam_names[0], '*.mp4')))

        # check if the 3d annotations exist 
        data_path = os.path.join(subject_path, f'save_data_AVG0_{subject}.mat')
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
                df = self.metadata[self.metadata['id'] == f'{subject}_{start_frame}']
                if not df['include'].values[0]: 
                    # print('skipping...')
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
                
                if subject == 'm4' and cam_name == 'Camera1':

                    for alt_video_path in alt_video_paths:

                        alt_video_name = os.path.basename(alt_video_path)
                        alt_start_frame = int(os.path.splitext(alt_video_name)[0])

                        video_info = io.deserialize_video(
                            alt_video_path, 
                            cam_outpath, 
                            start_frame = alt_start_frame, 
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
                'num_frames': max(n_frames), 
                'num_cameras': len(intrinsics)}

            # save camera metadata
            io.save_json(data = calib_dict, outpath = trial_outpath, 
                    fname = 'metadata.yaml')