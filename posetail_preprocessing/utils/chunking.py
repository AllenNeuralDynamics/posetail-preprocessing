import warnings
import numpy as np


def compute_frame_displacement(pose3d):
    """Per-frame mean keypoint displacement (NaN-aware), shape (T-1,).

    pose3d: (n_subjects, T, n_kpts, 3)
    """
    diff = np.diff(pose3d, axis=1)                       # (S, T-1, K, 3)
    disp = np.linalg.norm(diff, axis=-1)                 # (S, T-1, K)
    flat = disp.reshape(disp.shape[0], disp.shape[1], -1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        per_frame = np.nanmean(flat, axis=(0, 2))
    return np.nan_to_num(per_frame, nan=0.0)


def best_movement_window(pose3d, window_size, frame_start=0, frame_end=None):
    """Find the window_size-frame window with the highest cumulative displacement.

    Searches pose3d[:, frame_start:frame_end].
    Returns (best_absolute_start_frame, score).
    Falls back to (frame_start, 0.0) when not enough frames.
    """
    if frame_end is None:
        frame_end = pose3d.shape[1]

    sub = pose3d[:, frame_start:frame_end]
    L = sub.shape[1]

    if L < window_size:
        return frame_start, 0.0

    per_frame = compute_frame_displacement(sub)   # length L-1

    # sum of per_frame[i : i + window_size - 1] via prefix sums
    cs = np.concatenate([[0.0], np.cumsum(per_frame)])
    n_windows = len(per_frame) - (window_size - 1) + 1
    window_sums = cs[window_size - 1: window_size - 1 + n_windows] - cs[:n_windows]

    best_offset = int(np.argmax(window_sums))
    best_score = float(window_sums[best_offset])

    return frame_start + best_offset, best_score


def top_movement_windows(pose3d, window_size, n_windows,
                         frame_start=0, frame_end=None):
    """Return up to n_windows non-overlapping start frames (absolute),
    greedily ranked by cumulative displacement, within [frame_start, frame_end).

    Returns sorted list of absolute start frames.
    Falls back to [frame_start] when not enough frames.
    """
    if frame_end is None:
        frame_end = pose3d.shape[1]

    sub = pose3d[:, frame_start:frame_end]
    L = sub.shape[1]

    if L < window_size:
        return [frame_start]

    per_frame = compute_frame_displacement(sub)   # length L-1

    cs = np.concatenate([[0.0], np.cumsum(per_frame)])
    n_pos = len(per_frame) - (window_size - 1) + 1
    window_sums = cs[window_size - 1: window_size - 1 + n_pos] - cs[:n_pos]

    sums = window_sums.copy()
    starts = []

    for _ in range(n_windows):
        if np.all(sums == -np.inf):
            break
        best_offset = int(np.argmax(sums))
        starts.append(frame_start + best_offset)
        # mask out all overlapping windows
        lo = max(0, best_offset - window_size + 1)
        hi = min(n_pos, best_offset + window_size)
        sums[lo:hi] = -np.inf

    return sorted(starts)


def top_windows_across_segments(pose3d, window_size, n_windows, segments):
    """Select n_windows non-overlapping bout starts from a list of (start, end) segments,
    ranked globally by cumulative displacement.

    segments: list of (abs_start, abs_end) — already non-overlapping with each other.
    Returns sorted list of absolute start frames.
    """
    candidates = []   # (score, start)
    for seg_s, seg_e in segments:
        L = seg_e - seg_s
        if L < window_size:
            continue
        per_frame = compute_frame_displacement(pose3d[:, seg_s:seg_e])
        cs = np.concatenate([[0.0], np.cumsum(per_frame)])
        n_pos = L - 1 - (window_size - 1) + 1
        window_sums = cs[window_size - 1: window_size - 1 + n_pos] - cs[:n_pos]
        for offset, score in enumerate(window_sums):
            candidates.append((float(score), seg_s + offset))

    candidates.sort(reverse=True, key=lambda x: x[0])
    selected = []
    for score, start in candidates:
        if any(abs(start - s) < window_size for s in selected):
            continue
        selected.append(start)
        if len(selected) >= n_windows:
            break
    return sorted(selected)
