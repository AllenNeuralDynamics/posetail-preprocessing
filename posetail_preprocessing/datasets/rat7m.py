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
                 dataset_name = 'rat7m', debug_ix = None, 
                 filter_kernel_size = 11, filter_thresh = None, 
                 filter_percentile = 90):
        super().__init__(dataset_path, dataset_outpath)

        self.dataset_name = dataset_name
        self.metadata = None
        self.debug_ix = debug_ix

        # parameters for filtering ground truth keypoints
        self.kernel_size = filter_kernel_size
        self.thresh = filter_thresh
        self.percentile = filter_percentile
    
    def load_calibration(self, calib_path):

        mat = scipy.io.loadmat(calib_path)

        intrinsics_dict = {}
        extrinsics_dict = {}
        distortions_dict = {}
        sync_dict = {}

        cam_names = [x.lower() for x in mat['cameras'].dtype.names]
        # cam_order = [0, 1, 4, 2, 3, 5]

        for i, cam_name in enumerate(cam_names):

            params = mat['cameras'].item()[i]

            intrinsics = params['IntrinsicMatrix'].item().transpose()
            rotation_matrix = params['rotationMatrix'].item().transpose()
            tvec = params['translationVector'].item()

            extrinsics = assemble_extrinsics(rotation_matrix, tvec)
            # rvec = cv2.Rodrigues(rotation_matrix)[0].T[0]

            distortions = np.array([
                params['RadialDistortion'].item()[0,0],
                params['RadialDistortion'].item()[0,1],
                params['TangentialDistortion'].item()[0,0],
                params['TangentialDistortion'].item()[0,1], 0.0
            ])

            intrinsics_dict[cam_name] = intrinsics.tolist()
            extrinsics_dict[cam_name] = extrinsics.tolist()
            distortions_dict[cam_name] = distortions.tolist()

            # note the frames are mismatched in the raw dataset, so 
            # we have to index differently from the camera params
            sync_dict[cam_name] = params['frame'].item()[0] 

        return intrinsics_dict, extrinsics_dict, distortions_dict, sync_dict


    def load_pose3d(self, data_path):

        mat = scipy.io.loadmat(data_path)
        d = dict(zip(mat['mocap'].dtype.names, mat['mocap'].item()))
        bodyparts = mat['mocap'].dtype.names

        coords = []
        for bp in bodyparts:
            coords.append(d[bp])

        coords = np.array(coords)
        pose3d = np.expand_dims(coords.swapaxes(0, 1), axis = 0) # (n_subjects, time, bodyparts, 3)

        # TODO: improve filtering
        pose3d = pose3d[:, :, :8, :]
        bodyparts = bodyparts[:8]
        # pose3d = self._filter_coords(
        #     coords = pose3d, 
        #     kernel_size = self.kernel_size, 
        #     thresh = self.thresh, 
        #     percentile = self.percentile)
        
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

    def select_train_set(self, n_train_videos = 10, seed = 3):
        # randomly sample n training videos, distributed 
        # across training subjects

        np.random.seed(seed)

        # filter metadata for training videos
        self.metadata.loc[:, 'include'] = False
        self.metadata.loc[self.metadata['subject'] == 's5', 'split'] = 'test'
        train_df = self.metadata[self.metadata['split'] == 'train']

        # determine number of videos to sample from 
        # each group
        sessions = np.unique(train_df['session'])
        sample_dict = {session: 0 for session in sessions}
        n = 0

        while n < n_train_videos:

            for session in sessions: 
                sample_dict[session] += 1
                n += 1

                if n >= n_train_videos: 
                    break
                    
        # sample videos for training  
        train_ixs = []

        for session in sessions: 

            df_subset = self.metadata[self.metadata['session'] == session]
            n_to_sample = sample_dict[session]
            ixs = np.random.choice(df_subset.index, n_to_sample, replace = False)
            train_ixs.extend(ixs)   

        train_ixs = np.array(train_ixs)
        self.metadata.loc[train_ixs, 'include'] = True

        return self.metadata
    
    def select_train_set_single_session(self, session, n_train_videos = 10, seed = 3):
        # randomly sample n training videos from one subject
        # other than subject 5, which is the test set

        np.random.seed(seed)

        # filter metadata for training videos
        self.metadata.loc[:, 'include'] = False
        self.metadata.loc[self.metadata['subject'] == 's5', 'split'] = 'test'

        # sample videos for training  
        df_subset = self.metadata[self.metadata['session'] == session]
        train_ixs = np.random.choice(df_subset.index, n_train_videos, replace = False)
        self.metadata.loc[train_ixs, 'include'] = True

        return self.metadata

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

            # clean up any empty directories
            if len(os.listdir(outpath)) == 0:
                # print(f'removing: {outpath}')
                os.rmdir(outpath)
        

    def _filter_coords(coords, kernel_size = 11, thresh = None, percentile = 90): 
        ''' 
        filters rat7m coordinates by using a median filter to 
        detect outliar keypoints and masking them with nans 

        if thresh is none, will threshold according to a percentile
        for a given subject, keypoint, and coordinate (i.e. x, y, z)
        '''
        n_subjects, _, n_kpts, dim = coords.shape
        coords_filtered = np.zeros(coords.shape) 

        for i in range(n_subjects): 

            for j in range(n_kpts):

                for k in range(dim):

                    x = coords[i, :, j, k] # only one subject in this dataset
                    medfilt = scipy.signal.medfilt(x, kernel_size = kernel_size)
                    diff = np.abs(x - medfilt)
                    coords_filt = x.copy()

                    # use a percentile-based threshold if not provided an
                    # arbitrary threshold
                    if thresh is None: 
                        thresh = np.nanpercentile(diff, percentile)

                    coords_filt[diff >= thresh] = np.nan
                    coords_filtered[i, :, j, k] = coords_filt

        mask = np.isnan(coords_filtered).any(axis = -1)
        coords_filtered[mask] = np.nan

        return coords_filtered
    

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
                    'subject': session.split('-')[0], 
                    'trial': 1,
                    'n_cameras': n_cams, 
                    'n_frames': n_frames,
                    'total_frames': n_frames * n_cams,
                    'split': 'train',
                    'include': True}
        
            rows.append(metadata_dict)

        return rows
    
    def _process_session(self, calib_path, session_path, outpath, 
                         session, chunk_size = 3500): 

        # load calibration data
        calib_path = os.path.join(calib_path, f'mocap-{session}.mat')
        intrinsics, extrinsics, distortions, sync_dict = self.load_calibration(calib_path)

        video_paths = sorted(glob.glob(os.path.join(session_path, f'{session}-*.mp4')))

        cam_names = np.unique([os.path.splitext(os.path.basename(video_path))[0].split('-')[2]
                               for video_path in video_paths]).tolist()
        
        start_frames = np.unique([int(os.path.splitext(os.path.basename(video_path))[0].split('-')[3])
                    for video_path in video_paths])
        
        # load and format the 3d annotations
        pose_dict = self.load_pose3d(calib_path)
        pose = pose_dict['pose']

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
            pose_dict_subset = {'pose': pose[:, start_frame:start_frame + chunk_size, :, :], 
                                'keypoints': pose_dict['keypoints']}
            io.save_npz(pose_dict_subset, trial_outpath, fname = 'pose3d')

            # deserialize the camera videos and save as images 
            cam_height_dict = {}
            cam_width_dict = {}
            n_frames = []

            for cam_name in cam_names: 

                cam_frames = sync_dict[cam_name][start_frame:start_frame + chunk_size]

                for i, frame in enumerate(cam_frames): 

                    if i == self.debug_ix:
                        break

                    video_ix = frame // chunk_size
                    cam_start_frame = start_frames[video_ix]
                    cam_video_path = os.path.join(session_path, f'{session}-{cam_name}-{cam_start_frame}.mp4')
                    cam_outpath = os.path.join(trial_outpath, 'img', cam_name)

                    video_info = io.save_frame_synced(
                        video_path = cam_video_path, 
                        outpath = cam_outpath, 
                        frame_ix = frame - cam_start_frame, 
                        frame_ix_synced = i)

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