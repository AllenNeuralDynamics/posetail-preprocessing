import os
import cv2
import json
import json5
import toml
import yaml

import numpy as np
import subprocess

from decord import VideoReader, cpu

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


def get_video_info(video_path):

    cap = cv2.VideoCapture(video_path)

    video_info = {
        'camera_heights': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'camera_widths': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'num_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS)
    }

    cap.release()

    return video_info


def deserialize_video(video_path, outpath, start_frame = 0,
                      start_at = 0, 
                      debug_ix = None, zfill = 6):

    os.makedirs(outpath, exist_ok = True)
    cap = cv2.VideoCapture(video_path)

    video_info = {
        'camera_heights': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'camera_widths': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'num_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS)
    }

    print(video_path)
    print(start_at, debug_ix)

    
    frame_ix = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_ix < start_at:
            frame_ix += 1
            continue
        
        outname = str(frame_ix + start_frame - start_at).zfill(zfill) + '.jpg'
        out_path = os.path.join(outpath, outname)
        cv2.imwrite(out_path, frame)
        frame_ix += 1

        if debug_ix and frame_ix - start_at >= debug_ix: 
            break

    cap.release()

    return video_info

def deserialize_video_ffmpeg(video_path, outpath, start_number=0,
                             start_at=0, debug_ix=None, zfill=6):
    """NOTE: This is faster than deserialize_video
    BUT may not give as reliably synced frames for some videos (!!)"""
    
    os.makedirs(outpath, exist_ok=True)


    video_info = get_video_info(video_path)

    start_at_time = start_at / float(video_info['fps'])  
    
    
    # Build ffmpeg command
    ffmpeg_cmd = [
        'ffmpeg',
        '-hide_banner', '-loglevel', 'error', '-stats',
        '-i', video_path,
        '-ss', str(start_at_time),
        '-start_number', str(start_number),
        '-q:v', '1',
        '-vsync', '0'
    ]
    
    # Add frame limit if debug_ix is specified
    if debug_ix:
        ffmpeg_cmd.extend(['-vframes', str(debug_ix)])
    
    # Output pattern with zfill
    output_pattern = os.path.join(outpath, f'%0{zfill}d.jpg')
    ffmpeg_cmd.append(output_pattern)

    print(ffmpeg_cmd)
    
    # Run ffmpeg
    subprocess.run(ffmpeg_cmd, check=True, capture_output=False)
    
    return video_info



def deserialize_video_with_alignment(video_path, outpath, start_frame, 
                      end_frame, debug_ix = None, zfill = 6):

    os.makedirs(outpath, exist_ok = True)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    video_info = {
        'camera_heights': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'camera_widths': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'num_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': int(cap.get(cv2.CAP_PROP_FPS))
    }

    n_frames = end_frame - start_frame

    for i in range(n_frames): 

        ret, frame = cap.read()

        if not ret:
            break

        out_path = os.path.join(outpath, f'{str(i).zfill(zfill)}.jpg')
        cv2.imwrite(out_path, frame)

        if debug_ix and i + 1 == debug_ix: 
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
        'camera_heights': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'camera_widths': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'num_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 
        'fps': cap.get(cv2.CAP_PROP_FPS)
    }

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


def save_frames_decord(video_path, frame_indices, outpath,
                       frame_ix_synced=None, zfill=6):
    """Read frame_indices from video_path via decord and write as JPGs.

    frame_ix_synced: list of output filenames indices (defaults to 0, 1, 2, ...).
    Returns video_info dict like get_video_info().
    """
    os.makedirs(outpath, exist_ok=True)

    vr = VideoReader(video_path, ctx=cpu(0))

    video_info = {
        'camera_heights': vr[0].shape[0],
        'camera_widths': vr[0].shape[1],
        'num_frames': len(vr),
        'fps': vr.get_avg_fps()
    }

    if frame_ix_synced is None:
        frame_ix_synced = list(range(len(frame_indices)))

    # clamp to actual video length so callers don't need to know the exact frame count
    n_valid = len(vr)
    pairs = [(idx, syn) for idx, syn in zip(frame_indices, frame_ix_synced) if idx < n_valid]
    frame_indices = [p[0] for p in pairs]
    frame_ix_synced = [p[1] for p in pairs]

    frames = vr.get_batch(frame_indices).asnumpy()  # (N, H, W, 3) RGB

    for i, frame in enumerate(frames):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(outpath, f'{str(frame_ix_synced[i]).zfill(zfill)}.jpg')
        cv2.imwrite(out_path, frame_bgr)

    video_info['frames_written'] = len(frame_indices)
    return video_info


def save_frames_pyav(video_path, start_frame, n_frames, outpath, zfill=6,
                     progress=False, desc=None):
    """Extract ``n_frames`` consecutive frames starting at ``start_frame`` via PyAV.

    decord misbehaves on some mp4s (e.g. the Johnson-lab fly recordings), so this
    seek-and-decode helper mirrors the PyAV pattern in
    ``JARVIS-HybridNet/scripts/visualize_bouts.py``: seek to at/just-before the
    target frame, then decode forward, skipping until ``frame_idx >= start_frame``.
    This is frame-accurate and avoids decoding from frame 0 on multi-GB files.

    Writes JPGs named ``000000.jpg`` ... (relative to start_frame) and returns a
    ``video_info`` dict matching ``get_video_info`` plus ``frames_written``.

    Set ``progress=True`` for a per-frame tqdm bar (``desc`` sets its label).
    """
    import av
    from tqdm import tqdm

    os.makedirs(outpath, exist_ok=True)

    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'
    fps = float(stream.average_rate)
    time_base = float(stream.time_base)

    video_info = {
        'camera_heights': int(stream.height),
        'camera_widths': int(stream.width),
        'num_frames': int(stream.frames),
        'fps': fps,
    }

    # seek to at/just-before start_frame (lands on a keyframe)
    ts = int(start_frame / fps / time_base)
    container.seek(ts, stream=stream)

    pbar = tqdm(total=n_frames, desc=desc or 'frames', disable=not progress)

    written = 0
    for frame in container.decode(video=0):
        frame_idx = round(frame.pts * time_base * fps)
        if frame_idx < start_frame:
            continue
        if written >= n_frames:
            break
        frame_bgr = frame.to_ndarray(format='bgr24')
        out_name = str(written).zfill(zfill) + '.jpg'
        cv2.imwrite(os.path.join(outpath, out_name), frame_bgr)
        written += 1
        pbar.update(1)

    pbar.close()
    container.close()

    video_info['frames_written'] = written
    return video_info

