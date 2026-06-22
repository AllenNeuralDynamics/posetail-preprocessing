import warnings

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


def _nan_running_median(xy, win):
    '''nan-aware temporal running median of a (T, 2) trajectory.

    Returns (T, 2); a frame's median is over the centered window of half-width
    ``win // 2``, ignoring nans. Output is nan only where the whole window is
    nan. Vectorised (one ``nanmedian`` over a stacked window) so it stays fast
    on long recordings.
    '''
    T = xy.shape[0]
    half = win // 2
    pad = np.full((half, xy.shape[1]), np.nan, dtype = xy.dtype)
    xp = np.concatenate([pad, xy, pad], axis = 0)          # (T + 2*half, 2)
    stk = np.stack([xp[i:i + T] for i in range(win)], axis = 1)  # (T, win, 2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)    # all-nan windows
        med = np.nanmedian(stk, axis = 1)
    return med


def _prev_next_finite(fin):
    '''Index of nearest finite frame strictly before / after each frame.

    fin: (T,) bool. Returns (prev, nxt) int arrays; -1 where none exists. O(T).
    '''
    T = len(fin)
    prev = np.full(T, -1, dtype = int)
    last = -1
    for t in range(T):
        prev[t] = last
        if fin[t]:
            last = t
    nxt = np.full(T, -1, dtype = int)
    nextf = -1
    for t in range(T - 1, -1, -1):
        nxt[t] = nextf
        if fin[t]:
            nextf = t
    return prev, nxt


def despike_pose(pose, image_size, win = 5,
                 n_med = 4.0, floor_px = 6.0,
                 cap_n_med = 8.0, cap_floor_px = 20.0, max_frac = 0.06):
    '''
    removes implausible 2D-label jumps ("teleport" spikes) from a pose array by
    masking the offending points with nans. Built for the single-view,
    multi-individual 2D datasets (branson-fly, rat-city) whose raw tracking has a
    heavy tail of out-and-back label errors that the model cannot (and should
    not) fit.

    pose: (S, T, K, 2) float, raw pixels, nan = missing detection.

    Thresholds are ADAPTIVE per (subject, keypoint) trajectory, scaled to that
    track's own median frame-to-frame step ``ms`` -- crop/resolution invariant,
    which matters because the model crops tight around each animal, so what hurts
    is a jump large *relative to the animal*, not to the full frame. A spike
    threshold of ``max(floor_px, n_med * ms)`` flags a transient that is several
    times the track's normal motion; ``image_size`` only sets a ceiling of
    ``max_frac`` of the image diagonal so a degenerate (huge-``ms``) track can't
    disable cleaning.

    Three nan-aware passes per trajectory, each nan-ing the whole (s, t, k)
    point (mirrors ``mad_filter_coords``). All O(T):
      1. median-filter despike: nan frames whose euclidean distance from the
         local temporal running median (window ``win``) exceeds the spike
         threshold. Catches transient out-and-back spikes.
      2. out-and-back detector: nan p[t] when both immediate legs |p[t]-p[t-1]|
         and |p[t+1]-p[t]| exceed the spike threshold while the skip
         |p[t+1]-p[t-1]| stays below half the larger leg (a 1-frame excursion
         that snaps back -- an unambiguous label error).
      3. velocity cap: nan p[t] when the jump from BOTH the nearest finite frame
         before and after exceeds the (larger) cap threshold
         ``max(cap_floor_px, cap_n_med * ms)``. Catches multi-frame excursions
         the 1-frame detector misses, without clipping sustained fast motion
         (which stays consistent with >= 1 neighbour).

    Returns a cleaned copy of ``pose``.
    '''
    out = pose.copy()
    S, T, K, _ = pose.shape
    ceiling = float(max_frac * np.sqrt(2.0) * image_size)
    ar = np.arange(T)

    for s in range(S):
        for k in range(K):
            xy = out[s, :, k, :]                       # (T, 2) view -- mutate via out

            # per-track motion scale -> adaptive thresholds
            step = np.linalg.norm(np.diff(xy, axis = 0), axis = -1)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                ms = np.nanmedian(step)
            if not np.isfinite(ms):
                ms = 0.0
            spike_thr = min(max(floor_px, n_med * ms), ceiling)
            cap_thr = min(max(cap_floor_px, cap_n_med * ms), ceiling)

            # pass 1: median-filter despike
            med = _nan_running_median(xy, win)
            dist = np.linalg.norm(xy - med, axis = -1)
            xy[np.isfinite(dist) & (dist > spike_thr)] = np.nan

            # passes 2 & 3 operate on the post-pass-1 finite set
            fin = np.isfinite(xy[:, 0])
            if fin.sum() < 3:
                continue
            prev, nxt = _prev_next_finite(fin)
            hp = prev >= 0
            hn = nxt >= 0

            out_leg = np.full(T, np.nan)               # |p[t] - p[prev]|
            out_leg[hp] = np.linalg.norm(xy[ar[hp]] - xy[prev[hp]], axis = -1)
            in_leg = np.full(T, np.nan)                # |p[nxt] - p[t]|
            in_leg[hn] = np.linalg.norm(xy[nxt[hn]] - xy[ar[hn]], axis = -1)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                # pass 2: immediate-neighbour out-and-back spike
                immediate = fin & (prev == ar - 1) & (nxt == ar + 1)
                skip = np.full(T, np.inf)
                skip[immediate] = np.linalg.norm(
                    xy[nxt[immediate]] - xy[prev[immediate]], axis = -1)
                spike = (immediate & (out_leg > spike_thr) & (in_leg > spike_thr)
                         & (skip < 0.5 * np.maximum(out_leg, in_leg)))
                # pass 3: velocity cap (both finite neighbours far)
                cap = fin & hp & hn & (out_leg > cap_thr) & (in_leg > cap_thr)

            xy[spike | cap] = np.nan
            out[s, :, k, :] = xy

    # any nan coordinate nans the whole point
    mask = np.isnan(out).any(axis = -1)
    out[mask] = np.nan

    return out