import glob
import json
import os 
import cv2
import shutil

import numpy as np
import pandas as pd 

from tqdm import tqdm

from posetail_preprocessing.datasets import BaseDataset
from posetail_preprocessing.utils import io, assemble_extrinsics


class CMUPanopticDataset(BaseDataset): 

    def __init__(self, dataset_path, dataset_outpath, keypoints_path,
                 dataset_name = 'cmupanoptic'):
        super().__init__(dataset_path, dataset_outpath)

        self.dataset_name = dataset_name
        self.keypoints_path = keypoints_path    
    
    def load_calibration(self, calib_path):

        intrinsics_dict = {}
        extrinsics_dict = {}
        distortions_dict = {}
        resolution_dict = {}

        with open(calib_path) as f:
            data = json.load(f)

        cam_data = data['cameras']
        hd_cam_data = [d for d in cam_data if d['type'] == 'hd']

        for cam_data in hd_cam_data: 

            cam_name = str(int(cam_data['name'].split('_')[1]))

            tvec = np.array(cam_data['t'])
            rotation_matrix = cam_data['R']
            extrinsics = assemble_extrinsics(rotation_matrix, tvec)

            intrinsics_dict[cam_name] = cam_data['K']
            extrinsics_dict[cam_name] = extrinsics.tolist()
            distortions_dict[cam_name] = cam_data['distCoef']
            resolution_dict[cam_name] = cam_data['resolution']

        return intrinsics_dict, extrinsics_dict, distortions_dict, resolution_dict


    def load_pose3d(self, session_path):

        pose_kpts_path = os.path.join(self.keypoints_path, 'keypoints_pose_cmupanoptic.yaml')
        face_kpts_path = os.path.join(self.keypoints_path, 'keypoints_face_cmupanoptic.yaml')
        hand_kpts_path = os.path.join(self.keypoints_path, 'keypoints_hand_cmupanoptic.yaml')

        pose_kpts = io.load_yaml(pose_kpts_path)['keypoints']
        face_kpts = io.load_yaml(face_kpts_path)['keypoints']
        hand_kpts = io.load_yaml(hand_kpts_path)['keypoints']

        left_hand_kpts = ['l_' + kpt for kpt in hand_kpts]
        right_hand_kpts = ['r_' + kpt for kpt in hand_kpts]

        # aggregate 3d pose files
        n_kpts_pose = len(pose_kpts)
        pose_data_prefix = os.path.join(session_path, 'hdPose3d_stage1_coco19', 'hd')
        pose_data_paths = sorted(glob.glob(os.path.join(pose_data_prefix, '*.json')))
        ids_pose, n_frames_pose, start_frame_pose = self._get_unique_ids(pose_data_paths, kpt_type = 'pose')

        # aggregate 3d face files
        n_kpts_face = len(face_kpts)
        face_data_prefix = os.path.join(session_path, 'hdFace3d')
        face_data_paths = sorted(glob.glob(os.path.join(face_data_prefix, '*.json')))
        ids_face, n_frames_face, start_frame_face = self._get_unique_ids(face_data_paths, kpt_type = 'face')

        # aggregate 3d hand files
        n_kpts_hand = len(left_hand_kpts) + len(right_hand_kpts)
        hand_data_prefix = os.path.join(session_path, 'hdHand3d')
        hand_data_paths = sorted(glob.glob(os.path.join(hand_data_prefix, '*.json')))
        ids_hand, n_frames_hand, start_frame_hand = self._get_unique_ids(hand_data_paths, kpt_type = 'hand')

        # determine unique ids and frames, then populate 3d coords
        ids = ids_pose.union(ids_face, ids_hand)
        ids_to_index = dict(zip(ids, np.arange(len(ids))))

        # populate the coords for pose, face, and hand
        coords_pose = self._populate_coords(pose_data_paths, 
            ids_to_index, n_frames_pose, n_kpts_pose, kpt_type = 'pose')

        coords_face = self._populate_coords(face_data_paths, 
            ids_to_index, n_frames_face, n_kpts_face, kpt_type = 'face')

        coords_hand = self._populate_coords(hand_data_paths, 
            ids_to_index, n_frames_hand, n_kpts_hand, kpt_type = 'hand')

        # determine frame overlap to align temporally
        coords = [coords_pose, coords_face, coords_hand]
        start_frames = np.array([start_frame_pose, start_frame_face, start_frame_hand])
        n_frames = np.array([n_frames_pose, n_frames_face, n_frames_hand])

        coords_pose_aligned, coords_face_aligned, coords_hand_aligned = self._align_coords(coords, start_frames, n_frames)

        # combine body, face, and hand keypoints to construct pose dict
        subject_coords = np.concatenate((coords_pose_aligned, coords_face_aligned, coords_hand_aligned), axis = 2)
        kpts = pose_kpts + face_kpts + left_hand_kpts + right_hand_kpts
        pose3d_dict = {'pose': subject_coords, 'keypoints': kpts, 'ids': ids}

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

        self.split_frames_dict = split_frames_dict

        session_splits = [{'160906_pizza1'},  {'170915_office1'}, {'170407_office2'}]
        splits = ['val', 'test', None]

        for i, subjects in enumerate(session_splits):
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

            # skips sessions we aren't using
            if split is None: 
                continue 

            sessions = io.get_dirs(self.dataset_path)

            for session in tqdm(sessions, desc = split): 

                outpath = os.path.join(self.dataset_outpath, split, session, 'trial')
                os.makedirs(outpath, exist_ok = True)
                self._process_session(outpath, session, split)

                 # clean up any empty directories
                if len(os.listdir(outpath)) == 0:
                    # print(f'removing: {outpath}')
                    os.rmdir(outpath)


    def _get_start_frame(self, data_path): 

        base_name = os.path.splitext(os.path.basename(data_path))[0]
        start_frame = int(base_name.split('_')[1].lstrip('hd'))

        return start_frame


    def _get_unique_ids(self, data_paths, kpt_type = 'pose'):

        # determine whether paths are for the hand, 
        # face, or pose
        assert kpt_type in ['pose', 'face', 'hand']

        subject_key = 'people'
        if kpt_type == 'pose':
            subject_key = 'bodies'

        # find the unique ids and the number of frames
        start_frame = self._get_start_frame(data_paths[0])
        ids = set()

        for i, data_path in enumerate(data_paths): 

            data = io.load_json(data_path)
            bodies = data[subject_key]

            if len(bodies) == 0: 
                continue

            for body in bodies: 
                ids.add(body['id'])

        n_frames = i + 1

        return ids, n_frames, start_frame


    def _get_pose3d(self, subject, n_kpts, kpt_type):
        
        if kpt_type == 'pose': 
            pose = np.array(subject['joints19'])
            pose3d = pose.reshape(n_kpts, 4)[:, :3]

        elif kpt_type == 'face':
            pose = np.array(subject['face70']['landmarks'])
            pose3d = pose.reshape(n_kpts, 3)

        else: # kpt_type == 'hand' 
            assert n_kpts % 2 == 0
            
            if 'left_hand' in subject: 
                pose_left = np.array(subject['left_hand']['landmarks'])
            else: 
                pose_left = np.zeros(3 * n_kpts // 2) * np.nan

            if 'right_hand' in subject:
                pose_right = np.array(subject['right_hand']['landmarks'])
            else: 
                pose_right = np.zeros(3 * n_kpts // 2) * np.nan

            pose_left = pose_left.reshape(n_kpts // 2, 3)
            pose_right = pose_right.reshape(n_kpts // 2, 3)
            pose3d = np.vstack((pose_left, pose_right))

        return pose3d


    def _populate_coords(self, data_paths, ids_dict, n_frames, n_kpts, kpt_type = 'pose'):

        # determine whether paths are for the hand, 
        # face, or pose
        assert kpt_type in ['pose', 'face', 'hand']

        subject_key = 'people'
        if kpt_type == 'pose':
            subject_key = 'bodies'

        # populate the coords from each subject
        coords = np.zeros((len(ids_dict), n_frames, n_kpts, 3)) * np.nan

        for i, data_path in enumerate(data_paths): 

            data = io.load_json(data_path)
            subjects = data[subject_key]

            if len(subjects) == 0: 
                continue

            for subject in subjects: 
                id = subject['id']
                index = ids_dict[id]
                pose3d = self._get_pose3d(subject, n_kpts, kpt_type)
                coords[index, i, :, :] = pose3d 

        return coords


    def _align_coords(self, coords, start_frames, n_frames):

        end_frames = start_frames + n_frames
        common_start = np.max(start_frames)
        common_end = np.min(end_frames)
        common_n_frames = common_end - common_start
        offsets = common_start - start_frames

        coords_aligned = []

        for i, coords_subset in enumerate(coords): 

            offset = offsets[i]
            coords_subset = coords_subset[:, offset:offset + common_n_frames, :, :]
            coords_aligned.append(coords_subset)

        return coords_aligned
    

    def _get_sessions(self, session_path, session): 

        rows = []

        calib_path = os.path.join(session_path, f'calibration_{session}.json')
        intrinsics_dict, *_ = self.load_calibration(calib_path)
        n_cams = len(intrinsics_dict)

        # NOTE: will subsample cameras at dataset generation
        video_paths = sorted(glob.glob(os.path.join(session_path, 'hdVideos', '*.mp4')))
        pose_path = os.path.join(session_path, 'hdPose3d_stage1_coco19', 'hd')
        hand_path = os.path.join(session_path, 'hdHand3d')
        face_path = os.path.join(session_path, 'hdFace3d')

        if len(video_paths) == 0: 
            print(f'WARNING: missing videos for {session}')
            return rows
        
        if not any(os.path.exists(p) for p in (pose_path, hand_path, face_path)):
            print(f'WARNING: missing keypoint data for {session}')
            return rows

        cap = cv2.VideoCapture(video_paths[0])
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        metadata_dict = {
                'id': f'{session}',
                'session': session, 
                'subject':'', 
                # 'n_subjects': n_subjects,
                'trial': 1,
                'n_cameras': n_cams, 
                'n_frames': n_frames,
                'total_frames': n_frames * n_cams,
                'split': 'train',
                'include': True}
        
        rows.append(metadata_dict)

        return rows
        

    def _process_session(self, outpath, session, split): 

        # number of images to generate from each video
        split_frames = None
        if self.split_frames_dict and split in self.split_frames_dict: 
            split_frames = self.split_frames_dict[split]

        # select subset of metadata associated with the split 
        metadata = self.metadata[self.metadata['split'] == split]

        # specify conditions to process the session
        session_path = os.path.join(self.dataset_path, session)

        # load calibration data
        calib_path = os.path.join(session_path, f'calibration_{session}.json')
        intrinsics, extrinsics, distortions, _ = self.load_calibration(calib_path)
        cam_names = list(intrinsics.keys())

        # skip if metadata excludes it 
        process = True
        df = metadata[metadata['id'] == f'{session}']
        if df.empty or not df['include'].values[0]: 
            process = False

        if process: 
            # load and format the 3d annotations
            pose_dict = self.load_pose3d(session_path)
            pose_dict = self._subset_pose_dict(pose_dict, n_frames = split_frames)
            io.save_npz(pose_dict, outpath, fname = 'pose3d')

        # put videos/frames in the desired format
        if split == 'test':  
            # for test set, save as videos
            video_info = self._process_session_test(
                session_path, outpath, cam_names)
        else:
            # for train and validation sets, deserialize the camera videos 
            # and save as images  
            video_info = self._process_session_train(
                session_path, outpath, cam_names)

        cam_dict = {
            'intrinsic_matrices': intrinsics, 
            'extrinsic_matrices': extrinsics, 
            'distortion_matrices': distortions,
            'num_cameras': len(intrinsics)}
        cam_dict.update(video_info)

        # save camera metadata
        io.save_yaml(data = cam_dict, outpath = outpath, 
                fname = 'metadata.yaml')
        

    def _process_session_train(self, session_path, trial_outpath, cam_names,
                               split_frames = None):

        # copy image folders to new outpath
        cam_height_dict = {}
        cam_width_dict = {}
        num_frames = []
        fps = []

        for cam_name in cam_names:

            cam_video_path = os.path.join(session_path, 'hdVideos', f'hd_00_{str(cam_name).zfill(2)}.mp4')
            cam_outpath = os.path.join(trial_outpath, 'img', cam_name)

            video_info = io.deserialize_video(
                cam_video_path, 
                cam_outpath, 
                start_frame = 0, 
                debug_ix = split_frames)

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
    

    def _process_session_test(self, session_path, trial_outpath, cam_names):

        cam_height_dict = {}
        cam_width_dict = {}
        num_frames = []
        fps = []

        outpath = os.path.join(trial_outpath, 'vid')
        os.path.makedirs(outpath, exist_ok = True)

        for cam_name in cam_names: 

            cam_video_path = os.path.join(session_path, 'hdVideos', f'hd_{cam_name}.mp4')
            cam_video_outpath = os.path.join(outpath, f'{cam_name}.mp4')

            # extract info from the video     
            video_info = io.get_video_info(cam_video_path)
            cam_height_dict[cam_name] = video_info['camera_heights']
            cam_width_dict[cam_name] = video_info['camera_widths']
            num_frames.append(video_info['num_frames'])
            fps.append(video_info['fps'])

            # copy video to desired location
            os.symlink(cam_video_path, cam_video_outpath)

        video_info = {
            'cam_heights': cam_height_dict, 
            'cam_widths': cam_width_dict, 
            'num_frames': min(num_frames),
            'fps': min(fps)
        }

        return video_info