import glob
import os 
import cv2
import scipy

import numpy as np
import pandas as pd 

from tqdm import tqdm

from posetail_preprocessing.datasets import BaseDataset
from posetail_preprocessing.utils import io, assemble_extrinsics


class DukeMouseDataset(BaseDataset): 

    def __init__(self, dataset_path, dataset_outpath, 
                 dataset_name = 'duke_mouse'):
        super().__init__(dataset_path, dataset_outpath)

        self.dataset_name = dataset_name


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


    def select_splits(self, split_dict = None, split_frames_dict = None, 
                      random_state = 3):
        
        self.split_frames_dict = split_frames_dict

        subject_splits = [{'m1', 'm2'},  {'m3'},  {'m4', 'm5'}]
        splits = ['train', 'val', 'test']

        for i, subjects in enumerate(subject_splits):
            self.metadata.loc[self.metadata['subject'].isin(subjects), 'split'] = splits[i]

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

        # generate the dataset for each split 
        for split in splits: 

            subjects = io.get_dirs(self.dataset_path)

            for subject in tqdm(subjects, desc = split): 

                subject_path = os.path.join(self.dataset_path, subject)
                outpath = os.path.join(self.dataset_outpath, split, subject)
                os.makedirs(outpath, exist_ok = True)
                self._process_subject(subject_path, outpath, subject)

                # clean up any empty directories
                if len(os.listdir(outpath)) == 0:
                    os.rmdir(outpath)


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
    

    def _process_subject(self, subject_path, outpath, subject, 
                         split, chunk_size = 60000): 
        
        # number of images to generate from each video
        split_frames = None
        if self.split_frames_dict and split in self.split_frames_dict: 
            split_frames = self.split_frames_dict[split]

        # select the metadata for the given split
        metadata = self.metadata[self.metadata['split'] == split]

        # load calibration data
        calib_path = os.path.join(subject_path, 'annotations.mat')
        intrinsics, extrinsics, distortions = self.load_calibration(calib_path)

        video_dir = os.path.join(subject_path, 'videos')
        cam_names = io.get_dirs(video_dir) 
        video_paths = sorted(glob.glob(os.path.join(video_dir, cam_names[-1], '*.mp4')))

        # load and format the 3d annotations
        data_path = os.path.join(subject_path, f'save_data_AVG0_{subject}.mat')
        pose_dict = self.load_pose3d(data_path)
        pose = pose_dict['pose']

        # traverse the trials
        for video_path in video_paths: 

            # get starting frame
            video_name = os.path.basename(video_path)
            start_frame = int(os.path.splitext(video_name)[0])
            trial_outpath = os.path.join(outpath, str(start_frame))

            # skip video if metadata excludes it 
            if metadata is not None: 
                df = metadata[metadata['id'] == f'{subject}_{start_frame}']
                if df.empty or not df['include'].values[0]: 
                    # print('skipping...')
                    continue

            # load and format the 3d annotations
            if split_frames: 
                pose_dict_subset = {'pose': pose[start_frame: start_frame + split_frames, :, :], 
                                    'keypoints': pose_dict['keypoints']}
            else: 
                pose_dict_subset = {'pose': pose[start_frame: start_frame + chunk_size, :, :], 
                                    'keypoints': pose_dict['keypoints']}
                
            io.save_npz(pose_dict_subset, trial_outpath, fname = 'pose3d')

            # process trial depending on the split
            if split == 'test': 
                # for test set, save as videos
                video_info = self._process_subject_test(
                    video_dir, start_frame, cam_names, subject, 
                    trial_outpath)
            else: 
                # for train and validation sets, deserialize the camera videos 
                # and save as images  
                video_info = self._process_subject_train(
                    self, video_dir, start_frame, cam_names, 
                    subject, trial_outpath, split_frames = split_frames)

            calib_dict = {
                'intrinsic_matrices': intrinsics, 
                'extrinsic_matrices': extrinsics, 
                'distortion_matrices': distortions,
                'num_cameras': len(intrinsics)
            }
            calib_dict.update(video_info)

            # save camera metadata
            io.save_yaml(data = calib_dict, outpath = trial_outpath, 
                    fname = 'metadata.yaml')
            

    def _process_subject_train(self, video_dir, start_frame, cam_names, 
                               subject, trial_outpath, split_frames = None):
        
        # deserialize the camera videos and save as images 
        cam_height_dict = {}
        cam_width_dict = {}
        num_frames = []
        fps = []

        for cam_name in cam_names: 

            cam_video_path = os.path.join(video_dir, cam_name, f'{start_frame}.mp4')
            cam_outpath = os.path.join(trial_outpath, 'img', cam_name)

            video_info = io.deserialize_video(
                cam_video_path, 
                cam_outpath, 
                start_frame = 0, 
                debug_ix = split_frames)
            
            # this specific subject/camera pair have multiple videos recorded 
            # rather than just one like the other cameras, so it must be 
            # handled separately
            if subject == 'm4' and cam_name == 'Camera1':

                alt_video_paths = sorted(glob.glob(os.path.join(video_dir, 'Camera1', '*.mp4')))

                for alt_video_path in alt_video_paths:

                    alt_video_name = os.path.basename(alt_video_path)
                    alt_start_frame = int(os.path.splitext(alt_video_name)[0])

                    video_info = io.deserialize_video(
                        alt_video_path, 
                        cam_outpath, 
                        start_frame = alt_start_frame)

            cam_height_dict[cam_name] = video_info['camera_heights']
            cam_width_dict[cam_name] = video_info['camera_widths']
            num_frames.append(video_info['num_frames'])
            fps.append(video_info['fps'])

        video_info = {
            'cam_heights': cam_height_dict, 
            'cam_widths': cam_width_dict, 
            'num_frames': num_frames,
            'fps': fps
        }

        return video_info