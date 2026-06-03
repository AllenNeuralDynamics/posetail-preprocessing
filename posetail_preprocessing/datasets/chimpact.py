"""ChimpACTDataset — ChimpACT chimpanzee 2D pose (single-view, multi-individual).

ChimpACT (https://shirleymaxx.github.io/ChimpACT/) is a 2D-only dataset of zoo
chimpanzees. We emit the same flat per-trial layout as the ``branson-fly`` sibling
under ``posetail-finetuning-lili``::

    {out}/{split}/{clip_name}/ix000000/
        pose2d.npz               key 'pose', shape (S, T, 16, 2), float32, raw px, NaN=missing
        img/cam0/000000.jpg ...  one JPG per keyframe; img i <-> pose[:, i]

Key facts about the source release
(``/groups/branson/bransonlab/datasets/ChimpACT_release_v1``):
  * ``labels/*.json`` are per-clip COCO-style files. ChimpACT manually labels **1
    frame every 10**, so each clip has ~100 keyframes. The image ``file_name`` index
    ``k`` corresponds to **full-video frame ``k*10``** (see the official
    ``tools/create_coco_format.py``). Keypoints exist *only* at these keyframes.
  * Frames are extracted from ``videos_full/*.mp4`` (index-faithful to the labels).
  * ``bbox_id`` is the persistent per-clip track id -> the ``S`` (subjects) axis.
  * 16 keypoints, ``(x, y, v)`` per joint with ``v in {0, 1, 2}`` (0 = not labeled).

Splits follow the official ChimpACT membership (the hardcoded name lists below,
copied verbatim from ``tools/create_coco_format.py``): everything not in
``VAL_CLIPS``/``TEST_CLIPS`` is ``train``.
"""
import os
import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

from posetail_preprocessing.datasets import BaseDataset
from posetail_preprocessing.utils import io, compute_frame_displacement


# 16 keypoints, in the order stored in each clip's categories[0]['keypoints'].
KEYPOINTS = [
    'hip', 'r_knee', 'r_ankle', 'l_knee', 'l_ankle', 'neck',
    'upper_lip', 'lower_lip', 'r_eye', 'l_eye',
    'r_shoulder', 'r_elbow', 'r_wrist', 'l_shoulder', 'l_elbow', 'l_wrist',
]

# Official ChimpACT split membership (tools/create_coco_format.py). Train = the rest.
VAL_CLIPS = [
    'Azibo_ObsChimp_2015_11_25_d_clip_23000_24000',
    'Azibo_ObsChimp_2015_11_26_a_clip_1000_2000',
    'Azibo_ObsChimp_2016_08_02_c_clip_32000_33000',
    'Azibo_ObsChimp_2017_02_27_a_clip_13000_14000',
    'Azibo_ObsChimp_2017_11_10_clip_7000_8000',
    'Azibo_ObsChimp_2017_11_10_clip_8000_9000',
    'Azibo_ObsChimp_2017_06_22_c_clip_46000_47000',
    'Azibo_ObsChimp_2017_06_22_c_clip_67000_68000',
    'Azibo_ObsChimp_2018_07_11_c_clip_0_1000',
    'Azibo_ObsChimp_2018_07_11_c_clip_1000_2000',
    'Azibo_ObsChimp_2018_07_11_c_clip_3000_4000',
    'Azibo_ObsChimp_2018_07_11_c_clip_6000_7000',
    'Azibo_ObsChimp_2018_07_11_c_clip_17000_18000',
    'Azibo_ObsChimp_2018_07_11_c_clip_18000_19000',
    'Azibo_ObsChimp_2018_08_06_a_clip_7000_8000',
    'Azibo_ObsNatascha_2018_06_29_a_clip_15000_16000',
    'Azibo_ObsNatascha_2018_06_29_a_clip_16000_17000',
]

TEST_CLIPS = [
    'Azibo_ObsChimp_2015_11_25_d_clip_1000_2000',
    'Azibo_ObsChimp_2015_11_26_a_clip_0_1000',
    'Azibo_ObsChimp_2015_11_26_a_clip_2000_3000',
    'Azibo_ObsChimp_2016_08_02_c_clip_33000_34000',
    'Azibo_ObsChimp_2016_08_15_b_clip_2000_3000',
    'Azibo_ObsChimp_2016_10_27_c_clip_0_1000',
    'Azibo_ObsChimp_2017_02_27_a_clip_14000_15000',
    'Azibo_ObsChimp_2017_11_10_clip_6000_7000',
    'Azibo_ObsChimp_2017_06_22_c_clip_44000_45000',
    'Azibo_ObsChimp_2017_06_22_c_clip_68000_69000',
    'Azibo_ObsChimp_2018_07_06_d_clip_0_696',
    'Azibo_ObsChimp_2018_07_11_c_clip_2000_3000',
    'Azibo_ObsChimp_2018_07_11_c_clip_8000_9000',
    'Azibo_ObsChimp_2018_07_11_c_clip_16000_17000',
    'Azibo_ObsChimp_2018_07_11_c_clip_19000_20000',
    'Azibo_ObsChimp_2018_08_06_a_clip_6000_7000',
    'Azibo_ObsChimp_2018_08_06_a_clip_8000_9000',
    'Azibo_ObsNatascha_2018_06_29_a_clip_14000_15000',
    'Azibo_ObsNatascha_2018_06_29_a_clip_17000_17712',
]

FRAME_STRIDE = 10  # ChimpACT labels 1 frame in 10


class ChimpACTDataset(BaseDataset):

    def __init__(self, dataset_path, dataset_outpath, dataset_name='chimpact'):
        super().__init__(dataset_path, dataset_outpath)
        self.dataset_name = dataset_name
        self.labels_dir = os.path.join(dataset_path, 'labels')
        self.video_dir = os.path.join(dataset_path, 'videos_full')
        self.split_frames_dict = None

    # ------------------------------------------------------------------
    # BaseDataset abstract methods (2D-only -> no 3D / calibration)
    # ------------------------------------------------------------------

    def load_calibration(self, *args, **kwargs):
        return None

    def load_pose3d(self, *args, **kwargs):
        raise NotImplementedError('ChimpACT is a 2D-only dataset; use load_pose2d.')

    # ------------------------------------------------------------------
    # 2D pose loading
    # ------------------------------------------------------------------

    def _clip_split(self, clip_name):
        if clip_name in VAL_CLIPS:
            return 'val'
        if clip_name in TEST_CLIPS:
            return 'test'
        return 'train'

    def load_pose2d(self, clip_name):
        """Assemble a (S, T, 16, 2) pose array over a clip's keyframes.

        Returns dict with:
          pose:           (S, T, 16, 2) float32, raw px, NaN where v==0 / missing
          keypoints:      (16,) joint names
          frame_indices:  list[int] length T, full-video frame for each keyframe (k*10)
        """
        label_path = os.path.join(self.labels_dir, f'{clip_name}.json')
        data = io.load_json(label_path)

        # keyframe order: sort images by their file_name index (== keyframe k)
        def kf_idx(img):
            return int(os.path.splitext(os.path.basename(img['file_name']))[0])

        images = sorted(data['images'], key=kf_idx)
        T = len(images)
        image_id_to_t = {img['id']: t for t, img in enumerate(images)}
        frame_indices = [kf_idx(img) * FRAME_STRIDE for img in images]

        # subjects: distinct persistent track ids (bbox_id), sorted
        instance_ids = sorted({ann['bbox_id'] for ann in data['annotations']})
        inst_to_s = {iid: s for s, iid in enumerate(instance_ids)}
        S = len(instance_ids)

        n_kpts = len(KEYPOINTS)
        pose = np.full((S, T, n_kpts, 2), np.nan, dtype=np.float32)

        for ann in data['annotations']:
            t = image_id_to_t.get(ann['image_id'])
            if t is None:
                continue
            kpts = ann.get('keypoints', [])
            if len(kpts) != n_kpts * 3:
                continue
            s = inst_to_s[ann['bbox_id']]
            k = np.asarray(kpts, dtype=np.float32).reshape(n_kpts, 3)
            xy = k[:, :2]
            xy[k[:, 2] == 0] = np.nan  # v==0 -> not labeled
            pose[s, t] = xy

        return {
            'pose': pose,
            'keypoints': np.asarray(KEYPOINTS),
            'frame_indices': frame_indices,
        }

    @staticmethod
    def _drop_empty_subjects(pose):
        """Drop subjects (axis 0) that are all-NaN across the trial (cf. branson)."""
        keep = ~np.all(np.isnan(pose), axis=(1, 2, 3))
        return pose[keep]

    # ------------------------------------------------------------------
    # metadata / splits
    # ------------------------------------------------------------------

    def generate_metadata(self):
        """One row per clip: official split, keyframe count, movement score."""
        clip_paths = sorted(glob.glob(os.path.join(self.labels_dir, '*.json')))
        rows = []
        for lp in tqdm(clip_paths, desc='chimpact metadata'):
            clip_name = os.path.splitext(os.path.basename(lp))[0]
            pose = self._drop_empty_subjects(self.load_pose2d(clip_name)['pose'])
            T = pose.shape[1]
            score = float(np.sum(compute_frame_displacement(pose))) if T > 1 else 0.0
            rows.append({
                'id': clip_name,
                'clip_name': clip_name,
                'split': self._clip_split(clip_name),
                'n_subjects': int(pose.shape[0]),
                'n_keyframes': int(T),
                'movement_score': score,
                'include': True,
            })

        df = pd.DataFrame(rows)
        os.makedirs('metadata', exist_ok=True)
        df.to_csv(os.path.join('metadata', f'metadata_{self.dataset_name}.csv'),
                  index=False)
        self.metadata = df
        return df

    def select_splits(self, split_dict=None, split_frames_dict=None, random_state=3):
        """Sample a subset of clips per split; store per-split keyframe caps."""
        self.split_frames_dict = split_frames_dict or {}

        if split_dict:
            for split, n in split_dict.items():
                self._select_subset_for_split(split, n=n, random_state=random_state)

        return self.metadata

    # ------------------------------------------------------------------
    # main generation
    # ------------------------------------------------------------------

    def generate_dataset(self, splits=None):

        valid_splits = np.unique(self.metadata['split'])
        if splits is not None:
            splits = set(splits)
            assert splits.issubset(set(valid_splits)), (splits, valid_splits)
        else:
            splits = set(valid_splits)

        rows = self.metadata[self.metadata['split'].isin(splits)
                             & self.metadata['include']]

        for _, m in tqdm(list(rows.iterrows()), desc='chimpact trials'):
            clip_name = m['clip_name']
            split = m['split']

            pose_dict = self.load_pose2d(clip_name)
            pose = pose_dict['pose']
            frame_indices = pose_dict['frame_indices']

            # per-split keyframe cap (e.g. val -> first 32 keyframes)
            cap = (self.split_frames_dict or {}).get(split)
            if cap:
                pose = pose[:, :cap]
                frame_indices = frame_indices[:cap]

            pose = self._drop_empty_subjects(pose)
            if pose.shape[0] == 0 or pose.shape[1] == 0:
                print(f'  skipping {clip_name} ({split}): no usable keyframes')
                continue

            outpath = os.path.join(self.dataset_outpath, split, clip_name, 'ix000000')
            cam_outpath = os.path.join(outpath, 'img', 'cam0')

            video_path = os.path.join(self.video_dir, f'{clip_name}.mp4')
            info = io.save_frames_decord(video_path, frame_indices, cam_outpath)

            # keep pose aligned with the frames actually written (decord clamps
            # any index >= video length, which can only be the trailing ones)
            written = info.get('frames_written', len(frame_indices))
            if written < pose.shape[1]:
                pose = pose[:, :written]

            io.save_npz({'pose': pose.astype(np.float32)}, outpath, fname='pose2d')
