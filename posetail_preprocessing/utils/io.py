import os 
import cv2
import json
import json5
import toml
import yaml

import numpy as np


def get_dirs(path): 

    dirs = os.listdir(path)

    dirs = [d for d in dirs if os.path.isdir(os.path.join(path, d))
            and not d.startswith('.')]
    
    dirs = sorted(dirs) 
    
    return dirs


def load_json(path):
    ''' 
    loads data from a json file
    '''
    with open(path, 'r') as f:
        data = json.load(f)

    return data


def load_json5(path):
    ''' 
    loads data from a json file
    '''
    with open(path, 'r') as f:
        data = json5.load(f)

    return data


def load_yaml(path): 
    '''
    safely loads data from a yaml file
    '''
    with open(path, 'r') as f: 
        data = yaml.safe_load(f)

    return data


def save_json(data, outpath, fname):

    os.makedirs(outpath, exist_ok = True) 

    with open(os.path.join(outpath, fname), 'w') as json_file: 
        json.dump(data, json_file, indent = 1) 


def save_yaml(data, outpath, fname):

    os.makedirs(outpath, exist_ok = True) 

    with open(os.path.join(outpath, fname), 'w') as yaml_file: 
        yaml.dump(data, yaml_file) 


def save_npz(data, outpath, fname): 

    os.makedirs(outpath, exist_ok = True) 
    np.savez(os.path.join(outpath, fname), **data)


def write_keypoints_toml(keypoints, outdir, default_name = 'keypoints'):

    keypoints_dict = {'keypoints': keypoints}
    outpath = os.path.join(outdir, f'{default_name}.toml')

    with open(outpath, 'w') as f:
        toml.dump(keypoints_dict, f)


def deserialize_video(video_path, outpath, start_frame = 0, debug_ix = None, zfill = 6):

    os.makedirs(outpath, exist_ok = True)
    cap = cv2.VideoCapture(video_path)

    video_info = {
        'camera_height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'camera_width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'num_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}

    frame_ix = start_frame

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        out_path = os.path.join(outpath, f'{str(frame_ix).zfill(zfill)}.jpg')
        cv2.imwrite(out_path, frame)
        frame_ix += 1

        if debug_ix is not None and frame_ix - start_frame == debug_ix: 
            break

    cap.release()

    return video_info


def save_frame_synced(video_path, outpath, frame_ix, 
                      frame_ix_synced = None, zfill = 6):

    if frame_ix_synced is None: 
        frame_ix_synced = frame_ix 

    os.makedirs(outpath, exist_ok = True)
    cap = cv2.VideoCapture(video_path)

    video_info = {
        'camera_height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'camera_width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'num_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_ix)
    _, frame = cap.read()

    out_path = os.path.join(outpath, f'{str(frame_ix_synced).zfill(zfill)}.jpg')
    cv2.imwrite(out_path, frame)

    cap.release()

    return video_info


def get_frame_synced(video_path, frame_ix, 
                    frame_ix_synced = None):

    if frame_ix_synced is None: 
        frame_ix_synced = frame_ix 

    cap = cv2.VideoCapture(video_path)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_ix)
    _, frame = cap.read()

    cap.release()

    return frame