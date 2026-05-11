import glob
import json
import os 
import cv2

import numpy as np
import pandas as pd 

from itertools import chain
from tqdm import tqdm

from posetail_preprocessing.datasets import BaseDataset
from posetail_preprocessing.utils import io, assemble_extrinsics


class CMUPanopticDataset(BaseDataset): 

    def __init__(self, dataset_path, dataset_outpath, keypoints_path,
                 dataset_name = 'cmupanoptic', conf_thresh = None):
        super().__init__(dataset_path, dataset_outpath)

        self.dataset_name = dataset_name
        self.keypoints_path = keypoints_path 
        self.conf_thresh = conf_thresh # filters keypoints based on confidence threshold   
    
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

            cam_name = cam_data['name']

            tvec = np.array(cam_data['t'])
            rotation_matrix = cam_data['R']
            extrinsics = assemble_extrinsics(rotation_matrix, tvec)

            intrinsics_dict[cam_name] = cam_data['K']
            extrinsics_dict[cam_name] = extrinsics.tolist()
            distortions_dict[cam_name] = cam_data['distCoef']
            resolution_dict[cam_name] = cam_data['resolution']

        return intrinsics_dict, extrinsics_dict, distortions_dict, resolution_dict


    def load_pose3d(self, session_path):

        all_subject_ids = set()
        all_data_paths = []
        all_kpts = []
        kpt_types = []
        start_frames = []
        n_frames = []

        kpt_dict = {
            'pose': 'hdPose3d_stage1_coco19', 
            'hand': 'hdHand3d', 
            'face': 'hdFace3d'}

        for kpt_type, kpt_folder in kpt_dict.items(): 

            # check if kpt_type exists for the given session
            data_prefix = os.path.join(session_path, kpt_folder)
            if not os.path.exists(data_prefix): 
                continue

            kpts_path = os.path.join(self.keypoints_path, f'keypoints_{kpt_type}_cmupanoptic.yaml')
            kpts = io.load_yaml(kpts_path)['keypoints']
            if kpt_type == 'hand': 
                left_hand_kpts = ['l_' + kpt for kpt in kpts]
                right_hand_kpts = ['r_' + kpt for kpt in kpts]
                kpts = left_hand_kpts + right_hand_kpts

            # aggregate 3d files
            data_paths = sorted(glob.glob(os.path.join(data_prefix, '*.json')))
            if len(data_paths) == 0: 
                data_paths = sorted(glob.glob(os.path.join(data_prefix, 'hd', '*.json')))

            subject_ids, num_frames, start_frame = self._get_unique_ids(data_paths, kpt_type = kpt_type)

            all_subject_ids.update(subject_ids)
            all_data_paths.append(data_paths)
            all_kpts.append(kpts)
            kpt_types.append(kpt_type)
            start_frames.append(start_frame)
            n_frames.append(num_frames)

        start_frames = np.array(start_frames)
        n_frames = np.array(n_frames)

        # determine unique ids and frames, then populate 3d coords
        ids_to_ix = dict(zip(all_subject_ids, np.arange(len(all_subject_ids))))
        coords = []

        for i, kpt_type in enumerate(kpt_types):

            coords3d = self._populate_coords(all_data_paths[i], 
                ids_to_ix, n_frames[i], 
                n_kpts = len(all_kpts[i]), 
                kpt_type = kpt_type,
                conf_thresh = self.conf_thresh)

            coords.append(coords3d)
    
        # determine frame overlap to align temporally
        coords_aligned, common_start, common_end = self._align_coords(coords, start_frames, n_frames)

        # combine body, face, and hand keypoints to construct pose dict
        all_coords = np.concatenate((coords_aligned), axis = 2)
        all_kpts_flat = list(chain.from_iterable(all_kpts))
        pose3d_dict = {'pose': all_coords, 'keypoints': all_kpts_flat, 'ids': all_subject_ids, 
                       'start_frame': common_start, 'end_frame': common_end}

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

        session_splits = [{'160906_pizza1'}, {'170915_office1'}, {'170407_office2'}]
        splits = ['val', 'test', None]

        for i, sessions in enumerate(session_splits):
            self.metadata.loc[self.metadata['session'].isin(sessions), 'split'] = splits[i]

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
                self._process_session(outpath, session, split)


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


    def _get_pose3d(self, subject, n_kpts, kpt_type, conf_thresh = None):
        
        if kpt_type == 'pose': 
            pose = np.array(subject['joints19']).reshape(n_kpts, 4)
            pose3d = pose[:, :3]

            if conf_thresh: 
                conf = pose[:, 3]
                pose3d = pose3d[conf > conf_thresh]

        elif kpt_type == 'face':
            pose = np.array(subject['face70']['landmarks']).reshape(n_kpts, 3)
            pose3d = pose[:, :3]
            
            if conf_thresh: 
                conf = pose[:, 3]
                pose3d = pose3d[conf > conf_thresh]

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


    def _populate_coords(self, data_paths, ids_dict, n_frames, n_kpts, 
                         kpt_type = 'pose', conf_thresh = None):

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
                pose3d = self._get_pose3d(subject, n_kpts, kpt_type, conf_thresh = conf_thresh)
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

        return coords_aligned, common_start, common_end
    

    def _get_sessions(self, session_path, session): 

        rows = []

        calib_path = os.path.join(session_path, f'calibration_{session}.json')
        intrinsics_dict, *_ = self.load_calibration(calib_path)
        n_cams = len(intrinsics_dict)

        # NOTE: will subsample cameras at dataset generation
        video_paths = sorted(glob.glob(os.path.join(session_path, 'hdVideos', '*.mp4')))
        pose_path = os.path.join(session_path, 'hdPose3d_stage1_coco19')
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

        # extract cam names depending on the available videos 
        cam_videos = glob.glob(os.path.join(session_path, 'hdVideos', '*.mp4'))
        cam_names = [os.path.splitext(os.path.basename(cam_video))[0].split('_', 1)[1]
                     for cam_video in cam_videos]

        # skip if metadata excludes it 
        process = True
        df = metadata[metadata['id'] == f'{session}']
        if df.empty or not df['include'].values[0]: 
            process = False

        # skip if missing all annotations (24 sessions)
        pose_path_exists = os.path.exists(os.path.join(session_path, 'hdPose3d_stage1_coco19'))
        face_path_exists = os.path.exists(os.path.join(session_path, 'hdHand3d'))
        hand_path_exists = os.path.exists(os.path.join(session_path, 'hdFace3d'))
        
        if not pose_path_exists and not face_path_exists and not hand_path_exists:
            # print(f'no keypoint data for session {session}')
            process = False

        if process: 

            # process the session
            os.makedirs(outpath, exist_ok = True)

            # load and format the 3d annotations
            pose_dict = self.load_pose3d(session_path)
            common_start = pose_dict.pop('start_frame')
            common_end = pose_dict.pop('end_frame')
            pose_dict = self._subset_pose_dict(pose_dict, n_frames = split_frames)
            io.save_npz(pose_dict, outpath, fname = 'pose3d')

            # put videos/frames in the desired format
            if split == 'test':  
                # for test set, save as videos
                video_info = self._process_session_test(
                    session_path, outpath, cam_names, 
                    common_start, common_end)
            else:
                # for train and validation sets, deserialize the camera videos 
                # and save as images  
                video_info = self._process_session_train(
                    session_path, outpath, cam_names, 
                    split_frames, common_start, common_end)

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
                               split_frames = None, start_frame = None, end_frame = None):

        # copy image folders to new outpath
        cam_height_dict = {}
        cam_width_dict = {}
        num_frames = []
        fps = []

        for cam_name in cam_names:

            cam_video_path = os.path.join(session_path, 'hdVideos', f'hd_{cam_name}.mp4')
            cam_outpath = os.path.join(trial_outpath, 'img', cam_name)

            video_info = io.deserialize_video_with_alignment(
                cam_video_path, 
                cam_outpath, 
                start_frame = start_frame,
                end_frame = end_frame, 
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
    

    def _process_session_test(self, session_path, trial_outpath, cam_names, 
                              start_frame = None, end_frame = None):

        cam_height_dict = {}
        cam_width_dict = {}
        num_frames = []
        fps = []
        n_frames = end_frame - start_frame

        outpath = os.path.join(trial_outpath, 'vid')
        os.makedirs(outpath, exist_ok = True)

        for cam_name in cam_names: 

            cam_video_path = os.path.join(session_path, 'hdVideos', f'hd_{cam_name}.mp4')
            cam_video_outpath = os.path.join(outpath, f'{cam_name}.mp4')

            # extract info from the video     
            video_info = io.get_video_info(cam_video_path)
            cam_height_dict[cam_name] = video_info['camera_heights']
            cam_width_dict[cam_name] = video_info['camera_widths']
            num_frames.append(video_info['num_frames'])
            fps.append(video_info['fps'])

            # generate a subset of the video to match the given 
            # start and end frame (aligns with coordinates)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(cam_video_outpath, fourcc, video_info['fps'], 
                                    (video_info['camera_widths'], video_info['camera_heights']))

            cap = cv2.VideoCapture(cam_video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for i in range(n_frames): 

                ret, frame = cap.read()
                if not ret: 
                    break

                writer.write(frame)
            
            cap.release()
            writer.release()

        video_info = {
            'cam_heights': cam_height_dict, 
            'cam_widths': cam_width_dict, 
            'num_frames': min(num_frames),
            'fps': min(fps)
        }

        return video_info