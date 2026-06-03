import scipy

import numpy as np


def filter_coords(coords, kernel_size = 11, thresh = None, percentile = 90): 
    ''' 
    filters coordinates by using a median filter to 
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


def mad_filter_coords(coords, k = 6.0):
    '''
    masks temporal outlier keypoints with nans using a robust MAD test.

    for each subject / keypoint / coordinate, flags frames where the value
    deviates from the temporal median by more than ``k`` robust standard
    deviations (1.4826 * median-absolute-deviation). nan-aware; any coordinate
    flagged nans the whole (subject, frame, keypoint) point so it is dropped
    from the 3D pose (and therefore not rendered for review).
    '''
    out = coords.copy()
    n_subjects, _, n_kpts, dim = coords.shape

    for i in range(n_subjects):
        for j in range(n_kpts):
            for c in range(dim):
                x = out[i, :, j, c]
                med = np.nanmedian(x)
                mad = np.nanmedian(np.abs(x - med))
                if not np.isfinite(mad) or mad == 0:
                    continue
                bad = np.abs(x - med) > k * 1.4826 * mad
                out[i, bad, j, c] = np.nan

    mask = np.isnan(out).any(axis = -1)
    out[mask] = np.nan

    return out