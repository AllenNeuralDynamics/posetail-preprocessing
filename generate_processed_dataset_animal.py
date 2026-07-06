
import os 

import numpy as np 
import pandas as pd 

from posetail_preprocessing.datasets import (
    ZefDataset, 
    AcinosetDataset,
    AllenMouseDataset,
    AniposeFlyDataset,
    ChimpACTDataset,
    CMUPanopticDataset,
    CMUPanopticGSDataset,
    DexYCBDataset,
    JarvisMonkeyDataset,
    JohnsonMouseDataset,
    JohnsonFlyDataset,
    KubricMultiviewDataset,
    PairR24MDataset,
    POPDataset,
    Rat7MDataset,
    RatCityDataset,
    RavanFishDataset,
    SoberBirdDataset,
    VoigtsMouseDataset
)


def update_subsampling(splits, n_vids = 2, n_frames = 16): 

    split_dict = {}
    split_frames_dict = {}
    splits = set([split for split in splits if split != 'test'])

    for split in splits: 
        split_dict[split] = n_vids
        split_frames_dict[split] = n_frames

    return splits, split_dict, split_frames_dict


def generate_3dzef(prefix, out_prefix, dataset_name = '3dzef',
                   random_state = 3, debug = False):
    '''
    generates the preprocessed 3dzef (3DZeF20-pinhole) dataset

    Two-view (top + front) zebrafish, head keypoint only. Uses the source
    `train/` split (the only one with ground truth) and splits by sequence:
    train: ZebraFish-01/-02/-03 (all frames, 9888 frames/view)
    val:   ZebraFish-04, first 64 frames
    test:  ZebraFish-04, remaining 846 frames
    '''

    print(f'\ngenerating {dataset_name}...')
    dataset_path = os.path.join(prefix, '3dzef', '3DZeF20-pinhole')
    dataset_outpath = os.path.join(out_prefix, dataset_name)

    dataset = ZefDataset(
        dataset_path = dataset_path,
        dataset_outpath = dataset_outpath,
        dataset_name = dataset_name)

    df = dataset.generate_metadata()

    splits = {'train', 'val', 'test'}
    df = dataset.select_splits(random_state = random_state)

    dataset.generate_dataset(splits = splits, debug = debug)


def generate_acinoset(prefix, out_prefix, kpt_prefix, 
                      dataset_name = 'acinoset', 
                      random_state = 3, debug = False):
    
    '''
    generates the preprocessed acinoset dataset

    train: 20540 frames (all)
    val: 2 videos * 16  frames = 32 frames 
    test: 932 frames
    '''

    print(f'\ngenerating {dataset_name}...')
    dataset_path = os.path.join(prefix, dataset_name)
    dataset_outpath = os.path.join(out_prefix, dataset_name)
    keypoints_path = os.path.join(kpt_prefix, f'keypoints_{dataset_name}.yaml')

    dataset = AcinosetDataset(
        dataset_path = dataset_path, 
        dataset_outpath = dataset_outpath, 
        dataset_name = dataset_name,
        keypoints_path = keypoints_path)

    df = dataset.generate_metadata()

    # generate full training dataset (21k), full test data
    splits = {'train', 'val', 'test'}
    split_dict = {'train': None, 'val': 2} # number of videos to sample from the dataset
    split_frames_dict = {'train': None, 'val': 16} # number of consecutive frames per video to sample 

    if debug: 
        splits, split_dict, split_frames_dict = update_subsampling(splits)

    df = dataset.select_splits(
        split_dict = split_dict, 
        split_frames_dict = split_frames_dict, 
        random_state = random_state)

    dataset.generate_dataset(splits = splits)


def generate_anipose_fly(prefix, out_prefix, 
                         dataset_name = 'anipose_fly', 
                         random_state = 3, debug = False):

    ''' 
    generates the preprocessed anipose fly dataset
    '''

    print(f'\ngenerating {dataset_name}...')
    dataset_path = os.path.join(prefix, dataset_name)
    dataset_outpath = os.path.join(out_prefix, dataset_name)

    dataset = AniposeFlyDataset(
        dataset_path = dataset_path,
        dataset_outpath = dataset_outpath,
        dataset_name = dataset_name,
        error_thresh = 5,         # drop keypoints with reproj error >= this (px)
        conf_thresh = 0.7,        # drop keypoints with triangulation score < this
        min_ncams = 2,            # drop keypoints triangulated from < this many cams
        mad_k = 6.0,              # drop temporal MAD-outlier keypoints (robust stds)
        min_valid_frac = 0.60,    # gate: min fraction of keypoints surviving cleaning
                                  # (recalibrated for the cleaned pose; good subjects
                                  # sit ~0.63-0.74, noisy ones ~0.40-0.46)
        max_reproj_error = None,  # optional reprojection-error gate (px)
        # restrict to the single sarah/8.25.22 session (5 clean-tracking flies)
        subject_glob = 'sarah/8.25.22/Fly *',
        # hold out one fly; its bouts are split (trial-level) into val + test,
        # train = the other 4 flies
        holdout_subject = 'sarah/8.25.22/Fly 5_0')

    df = dataset.generate_metadata()

    # high-movement, quality-filtered bouts per trial (best window per trial,
    # survivors ranked by movement); bouts applied to test too
    splits = {'train', 'val', 'test'}
    # train (Fly 1-4): None = keep ALL quality-gated bouts (do NOT set a train
    # budget above the number that pass the gate, or floor-protection relaxes the
    # gate). Held-out Fly 5_0 is split by movement: top-2 -> val (~32 frames),
    # next 16 -> test (~256 frames). val/test frame windows MUST be equal.
    split_dict = {'train': None, 'val': 2, 'test': 16}          # bouts per split (None = all)
    split_frames_dict = {'train': 60, 'val': 16, 'test': 16}    # frames per bout

    if debug:
        split_dict = {'train': 2, 'val': 1, 'test': 2}
        split_frames_dict = {'train': 16, 'val': 8, 'test': 8}

    df = dataset.select_splits(
        split_dict = split_dict,
        split_frames_dict = split_frames_dict,
        random_state = random_state)

    dataset.generate_dataset(splits = splits)

def generate_sober_bird(prefix, out_prefix, 
                         dataset_name = 'sober-zebrafinch', 
                         random_state = 3, debug = False):

    ''' 
    generates the preprocessed anipose fly dataset
    '''

    print(f'\ngenerating {dataset_name}...')
    dataset_path = os.path.join(prefix, dataset_name)
    dataset_outpath = os.path.join(out_prefix, dataset_name)

    dataset = SoberBirdDataset(
        dataset_path = dataset_path, 
        dataset_outpath = dataset_outpath, 
        dataset_name = dataset_name)

    df = dataset.generate_metadata()

    # sample 60k training frames, full training dataset
    splits = {'train', 'val', 'test'}

    split_dict = {'train': 4, 'val': 1, 'test': 1} # number of videos to sample from the dataset
    split_frames_dict = {'train': 12000, 'val': 32, 'test': 12000} # number of consecutive frames per video to sample

    if debug: 
        splits, split_dict, split_frames_dict = update_subsampling(splits)

    df = dataset.select_splits(
        split_dict = split_dict, 
        split_frames_dict = split_frames_dict, 
        random_state = random_state)

    dataset.generate_dataset(splits = splits)

def generate_allen_mouse(prefix, out_prefix,
                         dataset_name = 'allen-mouse',
                         random_state = 3, debug = False):

    '''
    generates the preprocessed allen mouse dataset

    Single recording (motor-observatory_717764_2024-12-03_10-47-14).
    Chronological 80/10/10 split with multi-bout high-movement sampling.

    train: 120 bouts * 500 frames = 60k frames/view
    val:   1 bout   * 32 frames   = 32 frames/view
    test:  1 bout   * 500 frames  = 500 frames/view
    '''

    print(f'\ngenerating {dataset_name}...')
    dataset_path    = os.path.join(prefix, dataset_name)
    dataset_outpath = os.path.join(out_prefix, dataset_name)

    dataset = AllenMouseDataset(
        dataset_path    = dataset_path,
        dataset_outpath = dataset_outpath,
        dataset_name    = dataset_name,
        error_thresh    = 5,
        conf_thresh     = 0.7,
        quality_thresh  = 0.4)

    # n_bouts per split
    split_dict        = {'train': 120, 'val': 1, 'test': 1}
    # frames per bout
    split_frames_dict = {'train': 500, 'val': 32, 'test': 500}

    if debug:
        split_dict        = {'train': 2, 'val': 1, 'test': 1}
        split_frames_dict = {'train': 16, 'val': 8, 'test': 16}

    splits = list(split_dict.keys())

    dataset.generate_metadata()
    dataset.select_splits(
        split_dict        = split_dict,
        split_frames_dict = split_frames_dict,
        random_state      = random_state)

    dataset.generate_dataset(splits = splits)


def generate_johnson_mouse(prefix, out_prefix, 
                         dataset_name = 'johnson-mouse', 
                         random_state = 3, debug = False):

    ''' 
    generates the preprocessed johnson mouse dataset
    '''

    print(f'\ngenerating {dataset_name}...')
    dataset_path = os.path.join(prefix, dataset_name)
    dataset_outpath = os.path.join(out_prefix, dataset_name)

    dataset = JohnsonMouseDataset(
        dataset_path = dataset_path, 
        dataset_outpath = dataset_outpath, 
        dataset_name = dataset_name,
        conf_thresh = 0.5)

    dataset.generate_dataset()    

def generate_johnson_fly(out_prefix, dataset_name = 'johnson-fly',
                         random_state = 3, debug = False):

    '''
    generates the preprocessed johnson fly (walking) dataset

    Orthographic 7-camera rig. Bouts are detected from the predictor output,
    cropped (first/last 300 frames dropped), and split: every cropped bout
    except the last -> train (all frames); last bout -> val (first 64 frames)
    + test (remainder). Paths are absolute, so no `prefix` is needed.
    '''

    print(f'\ngenerating {dataset_name}...')

    data = [{
        "predictions": "/groups/johnson/johnsonlab/Elliott_Abe/fly50_predictions_Jan23/walking/fly50_pred_Jan23/Predictions_3D_20260124-143752",
        "recording": "/groups/johnson/johnsonlab/Elliott_Abe/FlyData/fly_walking/2025_10_12_15_06_46",
        "calibration": "/groups/johnson/johnsonlab/Elliott_Abe/FlyData/fly_walking/2025_10_12_15_06_46/calibration",
        "start": 0,
        "number": 86000,
    }]

    dataset_outpath = os.path.join(out_prefix, dataset_name)

    dataset = JohnsonFlyDataset(
        dataset_path = None,
        dataset_outpath = dataset_outpath,
        data = data,
        dataset_name = dataset_name)

    df = dataset.generate_metadata()

    # splits are fixed by bout detection inside generate_metadata
    df = dataset.select_splits(random_state = random_state)

    dataset.generate_dataset(splits = {'train', 'val', 'test'})


def generate_jarvis_monkey(prefix, out_prefix,
                         dataset_name = 'jarvis-monkey', 
                         random_state = 3, debug = False):

    ''' 
    generates the preprocessed jarvis monkey dataset
    '''

    print(f'\ngenerating {dataset_name}...')
    dataset_path = os.path.join(prefix, dataset_name)
    dataset_outpath = os.path.join(out_prefix, dataset_name)

    dataset = JarvisMonkeyDataset(
        dataset_path = dataset_path, 
        dataset_outpath = dataset_outpath, 
        dataset_name = dataset_name,
        conf_thresh = 0.5)

    dataset.generate_dataset()    

def generate_voigts_mouse(prefix, out_prefix, 
                         dataset_name = 'voigts-mouse', 
                         random_state = 3, debug = False):

    ''' 
    generates the preprocessed jarvis monkey dataset
    '''

    print(f'\ngenerating {dataset_name}...')
    dataset_path = os.path.join(prefix, dataset_name)
    dataset_outpath = os.path.join(out_prefix, dataset_name)

    dataset = VoigtsMouseDataset(
        dataset_path = dataset_path, 
        dataset_outpath = dataset_outpath, 
        dataset_name = dataset_name,
        conf_thresh = 0.5)

    dataset.generate_dataset()    
    
def generate_cmupanoptic(prefix, out_prefix, kpt_prefix, 
                         dataset_name = 'cmupanoptic', 
                         random_state = 3, debug = False): 

    ''' 
    generates the preprocessed cmupanoptic dataset
    '''

    print(f'\ngenerating {dataset_name}...')
    dataset_path = os.path.join(prefix, dataset_name)
    dataset_outpath = os.path.join(out_prefix, dataset_name)

    dataset = CMUPanopticDataset(
        dataset_path = dataset_path, 
        dataset_outpath = dataset_outpath, 
        dataset_name = dataset_name, 
        keypoints_path = kpt_prefix)

    df = dataset.generate_metadata()

    # sample 60k training frames
    splits = {'train', 'val', 'test'}
    split_dict = {'train': 3, 'val': 2} # number of videos to sample from the dataset
    split_frames_dict = {'train': 10, 'val': 16} # number of consecutive frames per video to sample 

    if debug: 
        splits, split_dict, split_frames_dict = update_subsampling(splits)

    df = dataset.select_splits(
        split_dict = split_dict, 
        split_frames_dict = split_frames_dict, 
        random_state = random_state)

    dataset.generate_dataset(splits = splits)


def generate_cmupanoptic3dgs(prefix, out_prefix, 
                             dataset_name = 'cmupanoptic_3dgs', 
                             random_state = 3): 
    ''' 
    generates the preprocessed cmupanoptic 3dgs dataset 

    train: None
    val: None
    test: 6 videos * 150 frames = 900 frames
    '''
    print(f'\ngenerating {dataset_name}...')

    dataset_path = os.path.join(prefix, 'panoptic-multiview')
    dataset_outpath = os.path.join(out_prefix, dataset_name)

    dataset = CMUPanopticGSDataset(
        dataset_path = dataset_path, 
        dataset_outpath = dataset_outpath, 
        dataset_name = dataset_name)

    df = dataset.generate_metadata()

    splits = {'test'}

    df = dataset.select_splits(
        random_state = random_state)

    # generate train and test splits
    dataset.generate_dataset(splits = splits)


def generate_dex_ycb(prefix, out_prefix, 
                     dataset_name = 'dex_ycb', 
                     random_state = 3): 
    ''' 
    generate prepocessed dex ycb dataset

    train: None
    val: None
    test: 10 videos * ~24 frames = 245 frames
    '''

    print(f'\ngenerating {dataset_name}...')
    dataset_path = os.path.join(prefix, 'dex-ycb-multiview')
    dataset_outpath = os.path.join(out_prefix, dataset_name)

    dataset = DexYCBDataset(
        dataset_path = dataset_path, 
        dataset_outpath = dataset_outpath, 
        dataset_name = dataset_name)

    df = dataset.generate_metadata()

    splits = {'test'}

    df = dataset.select_splits(
        random_state = random_state)
    
    # generate full dataset for testing 
    dataset.generate_dataset(splits = splits)


def generate_kubric_multiview(prefix, out_prefix, 
                              dataset_name = 'kubric-multiview', 
                              random_state = 3, debug = False):
    ''' 
    generate prepocessed dex ycb dataset

    train: 5000 videos * 24 frames = 120000 frames
    val: 2 videos * 24 frames = 48 frames
    test: 30 videos * 24 frames = 720 frames
    '''

    # generate full kubric multiview dataset for pretraining
    print(f'\ngenerating {dataset_name}...')
    dataset_path = os.path.join(prefix, dataset_name)
    dataset_outpath = os.path.join(out_prefix, dataset_name)

    dataset = KubricMultiviewDataset(
        dataset_path = dataset_path, 
        dataset_outpath = dataset_outpath, 
        dataset_name = dataset_name)

    df = dataset.generate_metadata()

    splits = {'train', 'val', 'test'}
    split_dict = {'val': 2}
    split_frames_dict = {'val': 24}

    if debug: 
        splits, split_dict, split_frames_dict = update_subsampling(splits)
    
    df = dataset.select_splits(
        split_dict = split_dict,
        split_frames_dict = split_frames_dict, 
        random_state = random_state)

    dataset.generate_dataset(splits = splits)


def generate_pairr24m(prefix, out_prefix, dataset_name = 'pair-r24m', 
                      random_state = 3, debug = False):
    ''' 
    generates the preprocessed pairr24m dataset

    train: 1225 videos * 49 frames = 60025 frames
    val: 2 videos * 16 frames = 32 frames
    test: 215910 frames
    '''
    print(f'\ngenerating {dataset_name}...')
    dataset_path = os.path.join(prefix, dataset_name)
    dataset_outpath = os.path.join(out_prefix, dataset_name)

    dataset = PairR24MDataset(
        dataset_path = dataset_path, 
        dataset_outpath = dataset_outpath, 
        dataset_name = dataset_name)

    df = dataset.generate_metadata()

    # sample 60k training frames, full training data
    splits = {'train', 'val', 'test'}
    split_dict = {'train': 1225, 'val': 2} # number of videos to sample from the dataset
    split_frames_dict = {'train': 49, 'val': 16} # number of consecutive frames per video to sample 

    if debug: 
        splits, split_dict, split_frames_dict = update_subsampling(splits)

    df = dataset.select_splits(
        split_dict = split_dict, 
        split_frames_dict = split_frames_dict, 
        random_state = random_state)

    dataset.generate_dataset(splits = splits)



def generate_3dpop(prefix, out_prefix, dataset_name = '3dpop', 
                   random_state = 3, debug = False): 
    ''' 
    generates the preprocessed 3dpop dataset

    train: 59 videos * 1017 frames = 60003 frames
    val: 2 videos * 16 frames = 32 frames
    test: 62901 frames
    '''
    print(f'\ngenerating {dataset_name}...')
    dataset_path = os.path.join(prefix, dataset_name)
    dataset_outpath = os.path.join(out_prefix, dataset_name)

    dataset = POPDataset(
        dataset_path = dataset_path, 
        dataset_outpath = dataset_outpath, 
        dataset_name = dataset_name)

    df = dataset.generate_metadata()

    # sample 60k training frames
    splits = {'train', 'val', 'test'}
    split_dict = {'train': 59, 'val': 2} # number of videos to sample from the dataset
    split_frames_dict = {'train': 1017, 'val': 16} # number of consecutive frames per video to sample 

    if debug: 
        splits, split_dict, split_frames_dict = update_subsampling(splits)

    df = dataset.select_splits(
        split_dict = split_dict, 
        split_frames_dict = split_frames_dict, 
        random_state = random_state)

    dataset.generate_dataset(splits = splits)


def generate_rat7m(prefix, out_prefix, dataset_name = 'rat7m', 
                   random_state = 3, debug = False): 

    ''' 
    generates the preprocessed rat7m dataset 

    train: 190 videos * 320 frames = 60800 frames 
    val: 2 videos * 16 frames = 32 frames 
    test: 130 videos * 3500 frames = 455000 frames
    '''

    print(f'\ngenerating {dataset_name}...')
    dataset_path = os.path.join(prefix, dataset_name)
    dataset_outpath = os.path.join(out_prefix, dataset_name)

    dataset = Rat7MDataset(
        dataset_path = dataset_path, 
        dataset_outpath = dataset_outpath, 
        dataset_name = dataset_name,
        filter_kernel_size = 11, 
        filter_thresh = None, 
        filter_percentile = 90) # TODO: maybe 95

    df = dataset.generate_metadata()

    # sample 60k training frames, generate full test set
    splits = {'train', 'val', 'test'}
    split_dict = {'train': 190, 'val': 2} # number of videos to sample from the dataset
    split_frames_dict = {'train': 320, 'val': 16} # number of consecutive frames per video to sample 

    if debug: 
        splits, split_dict, split_frames_dict = update_subsampling(splits)

    df = dataset.select_splits(
        split_dict = split_dict, 
        split_frames_dict = split_frames_dict, 
        random_state = random_state)

    dataset.generate_dataset(splits = splits)



def generate_chimpact(out_prefix, dataset_name = 'chimpact',
                      dataset_path = '/groups/branson/bransonlab/datasets/ChimpACT_release_v1',
                      random_state = 3, debug = False):

    '''
    generates the preprocessed ChimpACT dataset (2D-only, single-view, multi-individual)

    Splits follow the official ChimpACT membership; we then sample clips per split:
    train: all official-train clips (~127), full clips (~100 keyframes each)
    val:   1 official-val clip, capped to 32 keyframes
    test:  2 official-test clips, full clips

    output mirrors branson-fly:
        {out}/{split}/{clip}/ix000000/{pose2d.npz, img/cam0/*.jpg}
    '''

    print(f'\ngenerating {dataset_name}...')
    dataset_outpath = os.path.join(out_prefix, dataset_name)

    dataset = ChimpACTDataset(
        dataset_path = dataset_path,
        dataset_outpath = dataset_outpath,
        dataset_name = dataset_name)

    df = dataset.generate_metadata()

    splits = {'train', 'val', 'test'}
    split_dict = {'train': None, 'val': 1, 'test': 2} # number of clips to sample per split
    split_frames_dict = {'train': None, 'val': 32}    # keyframe cap per trial

    if debug:
        splits, split_dict, split_frames_dict = update_subsampling(splits)

    df = dataset.select_splits(
        split_dict = split_dict,
        split_frames_dict = split_frames_dict,
        random_state = random_state)

    dataset.generate_dataset(splits = splits)


def generate_rat_city(out_prefix, dataset_name='rat-city',
                      dataset_path='/groups/karashchuk/karashchuklab/animal-datasets/janelia-shared/rat-city/cohort7_20251209_1659',
                      random_state=3, debug=False):

    '''
    generates the preprocessed Rat City dataset (2D-only, single-view, multi-individual)

    A single continuous recording (12 rats, 4 keypoints). Chronological 80/10/10 split:
    train: full first 80% of frames as one contiguous bout
    val:   64 frames just after the train region
    test:  last 500 frames

    output mirrors chimpact:
        {out}/{split}/{clip}/ix{start}/{pose2d.npz, img/cam0/*.jpg}
    '''

    print(f'\ngenerating {dataset_name}...')
    dataset_outpath = os.path.join(out_prefix, dataset_name)

    dataset = RatCityDataset(
        dataset_path=dataset_path,
        dataset_outpath=dataset_outpath,
        dataset_name=dataset_name)

    dataset.generate_metadata()
    dataset.select_splits()
    dataset.generate_dataset(splits={'train', 'val', 'test'}, debug=debug)


def generate_ravan_fish(out_prefix, dataset_name='ravan-fish-sim',
                        dataset_path='/misc/public/forLili_fromAniket/dataset_for_Lili_20260511/test_Lili_dense_fish_max_20',
                        random_state=3, debug=False):

    '''
    generates the preprocessed Ravan fish (simulated dense zebrafish) dataset
    (2D-only, single-view, multi-individual)

    10 clips of crowded fish (up to 20), 12 keypoints. Splits by video (zef3d-style):
    train: zebrafish_000000.._000008, all 299 frames each
    val:   zebrafish_000009, frames [0, 32)
    test:  zebrafish_000009, frames [32, end)

    output mirrors chimpact / rat-city:
        {out}/{split}/{video}/ix{start}/{pose2d.npz, img/cam0/*.jpg}
    '''

    print(f'\ngenerating {dataset_name}...')
    dataset_outpath = os.path.join(out_prefix, dataset_name)

    dataset = RavanFishDataset(
        dataset_path=dataset_path,
        dataset_outpath=dataset_outpath,
        dataset_name=dataset_name)

    dataset.generate_metadata()
    dataset.select_splits()
    dataset.generate_dataset(splits={'train', 'val', 'test'}, debug=debug)


if __name__ == '__main__':

    # raw and processed data locations
    prefix = '/groups/karashchuk/karashchuklab/animal-datasets'
    # out_prefix = '/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning'
    # prefix = '/data/animal-datasets'
    out_prefix = '/data/animal-datasets-processed/posetail-finetuning-lili'

    os.makedirs(out_prefix, exist_ok = True)
    # kpt_prefix = '/home/ruppk2@hhmi.org/posetail-preprocessing/posetail_preprocessing/keypoints'

    # random state for reproducing which subsets of each
    # dataset are selected 
    random_state = 3
    debug = False # debugs on a small portion of the test and val sets

    # pretraining dataset 
    # generate_kubric_multiview(prefix, out_prefix, debug = debug)

    # finetuning datasets 
    # generate_acinoset(prefix, oust_prefix, kpt_prefix = kpt_prefix, random_state = random_state, debug = debug)
    # generate_anipose_fly(prefix, out_prefix, dataset_name='tuthill-fly', random_state = random_state, debug = debug)
    # generate_allen_mouse(prefix, out_prefix, random_state = random_state, debug = debug)
    # generate_chimpact(out_prefix, random_state = random_state, debug = debug)
    # generate_rat_city(out_prefix, random_state = random_state, debug = debug)
    # generate_ravan_fish(out_prefix, random_state = random_state, debug = debug)
    # generate_rat7m(prefix, out_prefix, random_state = random_state, debug = debug)
    # generate_pairr24m(prefix, out_prefix, random_state = random_state, debug = debug)
    generate_3dpop(prefix, out_prefix, random_state = random_state, debug = debug)
    #generate_3dzef(prefix, out_prefix, random_state = random_state, debug = debug)
    # generate_cmupanoptic(prefix, out_prefix, kpt_prefix = kpt_prefix, random_state = random_state, debug = debug)
    # generate_johnson_mouse(prefix, out_prefix, random_state = random_state, debug = debug)
    # generate_jarvis_monkey(prefix, out_prefix, random_state = random_state, debug = debug)
    # generate_johnson_fly(out_prefix, random_state = random_state, debug = debug)
    # generate_allen_mouse(prefix, out_prefix, random_state = random_state, debug = debug)
    # generate_voigts_mouse(prefix, out_prefix, random_state = random_state, debug = debug)
    # generate_sober_bird(prefix, out_prefix, random_state = random_state, debug = debug)
    
    # purely test datasets
    # generate_cmupanoptic3dgs(prefix, out_prefix, random_state = random_state)
    # generate_dex_ycb(prefix, out_prefix, random_state = random_state) 
