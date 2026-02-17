import glob
import os 
import cv2
import toml
import shutil

import numpy as np
import pandas as pd 

from einops import rearrange
from tqdm import tqdm

from posetail_preprocessing.datasets import BaseDataset
from posetail_preprocessing.utils import io, assemble_extrinsics

from aniposelib.cameras import CameraGroup, Camera

import re

def get_mat_from_file(filepath, nodeName):
    fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
    return fs.getNode(nodeName).mat()

def true_basename(fname):
    basename = os.path.basename(fname)
    basename = os.path.splitext(basename)[0]
    return basename

def camera_from_path(calib_path):
    basename = os.path.basename(calib_path)
    basename, _ = os.path.splitext(basename)
    tvec = get_mat_from_file(calib_path, 'T')
    # note this has a transposed intrinsics and rotation matrix for some reason
    rvec = cv2.Rodrigues(get_mat_from_file(calib_path, 'R').T)[0]
    mat = get_mat_from_file(calib_path, 'intrinsicMatrix').T
    dist = get_mat_from_file(calib_path, 'distortionCoefficients')
    cam = Camera(matrix=mat, rvec=rvec, tvec=tvec, dist=dist, name=basename)
    return cam

class JarvisMonkeyDataset(BaseDataset): 

    def __init__(self, dataset_path, dataset_outpath, 
                 dataset_name = 'jarvis-monkey', conf_thresh = 0.2):
        super().__init__(dataset_path, dataset_outpath)

        self.dataset_name = dataset_name
        self.conf_thresh = conf_thresh

    def load_calibration(self, calib_fnames):

        intrinsics_dict = {}
        extrinsics_dict = {}
        distortions_dict = {}
        
        for fname in calib_fnames:
            cam = camera_from_path(fname)
            cname = cam.get_name()
            intrinsics_dict[cname] = cam.get_camera_matrix().tolist()
            extrinsics_dict[cname] = cam.get_extrinsics_mat().tolist()
            distortions_dict[cname] = cam.get_distortions().tolist()
        
        return intrinsics_dict, extrinsics_dict, distortions_dict


    def load_pose3d(self, data_path):

        df = pd.read_csv(data_path, header=[0,1])

        unique_kpts = np.unique(df.columns.levels[0])

        coords = df.loc[:, (unique_kpts, ['x', 'y', 'z'])].values
        n_frames = coords.shape[0]
        n_kpts = len(unique_kpts)
        coords = rearrange(coords, 't (n r) -> t n r', t = n_frames, n = n_kpts)

        
        # filter coords
        if self.conf_thresh:
            confs = df.loc[:, (unique_kpts, 'confidence')].values
            conf_mask = (confs < self.conf_thresh)
            coords[conf_mask] = np.nan

        # undo transformation
        coords = rearrange(coords, 't n r -> (t n) r', t = n_frames, n = n_kpts)

        pose3d = rearrange(coords, '(t n) r -> 1 t n r', t = n_frames, r = 3)  # (n_subjects, time, kpts, 3)
        pose3d_dict = {'pose': pose3d, 'keypoints': unique_kpts}

        return pose3d_dict

    def generate_metadata(self):
        pass

    def select_splits(self, split_dict = None, split_frames_dict = None, 
                      random_state = 3):
        pass # everything happens in generate_dataset
    
    def generate_dataset(self, splits = None): 

        pose_path = os.path.join(self.dataset_path, 'predictions', 'predictions3D',
                                 'Predictions_3D_20260125-085654', 'data3D.csv')
        pose_dict = self.load_pose3d(pose_path)

        trial = 'Example_Recording'
        calib_fnames = sorted(glob.glob(os.path.join(self.dataset_path, 'calib_params', '12Cam_Ralph', '*.yaml')))
        video_fnames = sorted(glob.glob(os.path.join(self.dataset_path, trial, '*.mp4')))

        intrinsics, extrinsics, distortions = self.load_calibration(calib_fnames)
        
        splits = ["train", "val", "test"]
        split_dict = {'train': 0.5, 'test': 0.05}

        cam_height_dict = {}
        cam_width_dict = {}
        num_frames = []
        fps = []
        
        for vidname in video_fnames:
            video_info = io.get_video_info(vidname)
            cname = true_basename(vidname)
            cam_height_dict[cname] = video_info['camera_heights'] 
            cam_width_dict[cname] = video_info['camera_widths']
            num_frames.append(video_info['num_frames'])
            fps.append(video_info['fps'])

        calib_dict = {
            'camera_heights': cam_height_dict, 
            'camera_widths': cam_width_dict, 
            'num_frames': min(min(num_frames), pose_dict['pose'].shape[1]),
            'fps': min(fps),
            'intrinsic_matrices': intrinsics, 
            'extrinsic_matrices': extrinsics, 
            'distortion_matrices': distortions,
            'num_cameras': len(intrinsics)
        }
        
        
        nframes = calib_dict['num_frames']
        print(nframes)
                      
        split_start_frames = {'train': 0, 'val': int(split_dict['train']*nframes + 50),
                              'test': nframes - int(split_dict['test'] * nframes)}
        split_num_frames = {'train': int(split_dict['train']*nframes),
                            'val': 32,
                            'test': int(split_dict['test']*nframes)-50}

        
        os.chdir(self.dataset_path)
            
        # generate the dataset for each split
        for split in splits:
            print(split)
            start_frame = split_start_frames[split]
            num_frames = split_num_frames[split]
            
            pose_dict_subset = self._subset_pose_dict(
                dict(pose_dict), start_frame = start_frame, n_frames = num_frames)

            outpath = os.path.join(self.dataset_outpath, split, 'Ralph',
                                   '{}_ix{}'.format(trial, start_frame))
            
            for cam_video_path in video_fnames:
                cam_name = true_basename(cam_video_path)
                # deserialize video into images
                cam_outpath = os.path.join(outpath, 'img', cam_name)
                os.makedirs(cam_outpath, exist_ok = True)
        
                io.deserialize_video_ffmpeg(
                    cam_video_path, cam_outpath,
                    start_at = start_frame, debug_ix = num_frames)
            
            io.save_npz(pose_dict_subset, outpath, fname = 'pose3d')
            io.save_yaml(data = calib_dict, outpath = outpath, fname = 'metadata.yaml')







