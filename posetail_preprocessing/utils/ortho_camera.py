"""Closed-form orthographic -> pinhole camera construction.

Johnson-lab fly recordings are calibrated with 11-coefficient DLT files
(``Cam*_dlt.csv``) whose last three coefficients are zero, i.e. the cameras
are orthographic. The main ``posetail`` repo only reads the pinhole convention
(``intrinsic_matrices`` / ``extrinsic_matrices`` / ``distortion_matrices``), so
this module rewrites each ortho DLT as a "telephoto pinhole" (K, ext) pair that
approximates the exact ortho projection to within 0.5 px on the recording's GT
keypoints, with zero distortion.

The construction follows ``preprocess.md`` (Steps 1-3 + validation tests 1-4)
and is a direct port of the validated reference implementation in
``/tmp/claude/ortho_tests/check_preprocess.py`` (which passes all checks on the
fly-walking recording).
"""
import os
import glob
import math

import numpy as np


EPS_PIXEL = 0.5       # sub-pixel honesty target for the pinhole approximation
MAD_K = 10.0          # robust filter: keep points within median +- K * MAD
DLT_AB_SCALE = 10.0   # divide L0..L2, L4..L6 by this (JARVIS convention; L3, L7 untouched)


def read_dlt(path):
    """Read an 11-coefficient DLT file into a flat (11,) float array."""
    return np.loadtxt(path).astype(np.float64).reshape(-1)


def cam_name_from_dlt(path):
    return os.path.basename(path).replace('_dlt.csv', '')


def read_image_size(mp4_path):
    """Return (W, H) for an mp4 via OpenCV."""
    import cv2
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f'cannot open {mp4_path}')
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return W, H


def load_dlt_cameras(calib_dir, recording_dir, scale=DLT_AB_SCALE):
    """Load every ``Cam*_dlt.csv`` and build the per-camera projection.

    Returns a list of cam dicts with keys ``name, L, proj_mat, A, t_proj, W, H``.
    Asserts each file is orthographic (``L[8:11] == 0``).
    """
    dlt_paths = sorted(glob.glob(os.path.join(calib_dir, 'Cam*_dlt.csv')))
    if not dlt_paths:
        raise FileNotFoundError(f'no Cam*_dlt.csv files found in {calib_dir}')
    cams = []
    for p in dlt_paths:
        name = cam_name_from_dlt(p)
        L = read_dlt(p)
        if L.shape != (11,):
            raise ValueError(f'{p}: expected 11 coefficients, got {L.shape}')
        # Confirm orthographic: L9 = L10 = L11 = 0
        if not (abs(L[8]) < 1e-9 and abs(L[9]) < 1e-9 and abs(L[10]) < 1e-9):
            raise ValueError(
                f'{name}: L[8..10] not zero (={L[8:11]}); not orthographic')
        # JARVIS convention: divide L0..L2 and L4..L6 by scale; L3, L7 unchanged.
        s = scale
        proj_mat = np.array([
            [L[0] / s, L[1] / s, L[2] / s, L[3]],
            [L[4] / s, L[5] / s, L[6] / s, L[7]],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        A = proj_mat[:2, :3]
        t_proj = proj_mat[:2, 3]
        mp4 = os.path.join(recording_dir, f'{name}.mp4')
        if os.path.exists(mp4):
            W, H = read_image_size(mp4)
        else:
            W, H = None, None
        cams.append({
            'name': name,
            'L': L,
            'proj_mat': proj_mat,
            'A': A,
            't_proj': t_proj,
            'W': W,
            'H': H,
        })
    return cams


def robust_filter(X, k=MAD_K):
    """Drop points whose any coord is more than k * MAD from the per-coord median.

    Real prediction CSVs include failed-triangulation outliers far outside the
    scene; without this filter a single outlier dominates the worst-case in the
    sub-pixel test and inflates D* by orders of magnitude.

    Returns (X_kept, n_dropped, (lo, hi)).
    """
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    mad = np.where(mad < 1e-12, 1.0, mad)  # avoid degenerate mad=0
    lo = med - k * mad
    hi = med + k * mad
    mask = ((X >= lo) & (X <= hi)).all(axis=1)
    return X[mask], int((~mask).sum()), (lo.tolist(), hi.tolist())


def orient_proj_dir(A, X):
    """preprocess.md Step 1: pd = cross(A[0], A[1]), oriented by the data."""
    raw_pd = np.cross(A[0], A[1])
    pd = raw_pd / np.linalg.norm(raw_pd)
    if np.median(X @ pd) < 0:
        pd = -pd
    return pd


def fit_D_star(cams, X, eps=EPS_PIXEL, margin_rel=0.1):
    """preprocess.md Step 2: pick the common rig distance D* (sub-pixel honest).

    Mutates each cam dict with ``A_pinv`` and ``sigma_max_A``. Returns
    ``(D_star, info_dict)``.
    """
    D_pinhole_per_cam = []
    D_positivity_per_cam = []
    scene_extent_along_pd = []
    for cam in cams:
        A = cam['A']
        pd = cam['pd']
        AAT = A @ A.T
        A_pinv = A.T @ np.linalg.inv(AAT)  # (3, 2)
        origins = (A_pinv @ (A @ X.T)).T   # (N, 3)
        s_vals = X @ pd                    # (N,)
        origins_norm = np.linalg.norm(origins, axis=1)
        sigma_max_A = np.linalg.svd(A, compute_uv=False)[0]
        worst = float((origins_norm * np.abs(s_vals)).max())
        D_pinhole_per_cam.append((2.0 / eps) * sigma_max_A * worst)
        D_positivity_per_cam.append(float(-s_vals.min()))
        scene_extent_along_pd.append(float(s_vals.max() - s_vals.min()))
        cam['A_pinv'] = A_pinv
        cam['sigma_max_A'] = sigma_max_A
    D_pinhole = max(D_pinhole_per_cam)
    D_positivity = max(D_positivity_per_cam)
    small_margin = margin_rel * max(scene_extent_along_pd)
    D_star = max(D_pinhole, D_positivity + small_margin)
    return D_star, {
        'D_pinhole': D_pinhole,
        'D_positivity': D_positivity,
        'small_margin': small_margin,
        'per_cam_D_pinhole': D_pinhole_per_cam,
    }


def build_rt_K(cam, D_star):
    """preprocess.md Step 3: build (R, t, ext, K) for one camera.

    Chooses ``u_z = +-pd`` so that ``fx > 0`` AND ``det(R) = +1``. Returns
    ``(R, t, ext, K, z_branch)``.
    """
    A = cam['A']
    pd = cam['pd']
    t_proj = cam['t_proj']
    u_y = A[1] / np.linalg.norm(A[1])
    u_x_proper = np.cross(u_y, pd)             # candidate if u_z = +pd
    if float(A[0] @ u_x_proper) >= 0:
        u_z = pd
        u_x = u_x_proper
        C = -D_star * pd
        z_branch = '+pd'
    else:
        u_z = -pd
        u_x = -u_x_proper                      # = u_y x u_z
        C = D_star * pd
        z_branch = '-pd'
    R = np.stack([u_x, u_y, u_z], axis=0)
    t = -R @ C
    ext = np.eye(4, dtype=np.float64)
    ext[:3, :3] = R
    ext[:3, 3] = t
    fx = D_star * float(A[0] @ u_x)            # > 0 by construction
    fy = D_star * np.linalg.norm(A[1])         # > 0 always
    skew = D_star * float(A[0] @ u_y)          # K[0, 1]
    cx = t_proj[0]
    cy = t_proj[1]
    K = np.array([[fx, skew, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return R, t, ext, K, z_branch


def project_pinhole(K, ext, X):
    """Project (N, 3) world points through K @ (R X + t). Returns (N, 2)."""
    Xh = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    p_cam = (ext @ Xh.T).T[:, :3]
    pix = (K @ p_cam.T).T
    return pix[:, :2] / pix[:, 2:3]


def project_dlt(A, t_proj, X):
    """Exact ortho DLT projection of (N, 3) world points. Returns (N, 2)."""
    return (A @ X.T).T + t_proj


def run_validations(cams, X, D_star, eps=EPS_PIXEL):
    """Run preprocess.md validation tests 1-4, building (R, t, ext, K) per cam.

    Mutates each cam with ``R, t, ext, K, z_branch``. Raises ``ValueError`` with
    a clear message naming the camera and worst keypoint on any failure.
    """
    sub_pixel = {'worst': 0.0, 'cam': None, 'pt': None}
    pos_depth = {'worst': math.inf, 'cam': None, 'pt': None}
    se3 = {'worst_orth': 0.0, 'min_det': 1.0, 'cam': None}
    k_pos = {'fx_min': math.inf, 'fy_min': math.inf, 'cam': None}

    for cam in cams:
        R, t, ext, K, z_branch = build_rt_K(cam, D_star)
        cam['R'], cam['t'], cam['ext'], cam['K'] = R, t, ext, K
        cam['z_branch'] = z_branch

        # Test 1: sub-pixel honesty
        p_pin = project_pinhole(K, ext, X)
        p_dlt = project_dlt(cam['A'], cam['t_proj'], X)
        err = np.linalg.norm(p_pin - p_dlt, axis=1)
        idx = int(err.argmax())
        if err[idx] > sub_pixel['worst']:
            sub_pixel.update(worst=float(err[idx]), cam=cam['name'],
                             pt=X[idx].tolist())

        # Test 2: positive depth
        Xh = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        Z = (ext @ Xh.T).T[:, 2]
        idx2 = int(Z.argmin())
        if Z[idx2] < pos_depth['worst']:
            pos_depth.update(worst=float(Z[idx2]), cam=cam['name'],
                             pt=X[idx2].tolist())

        # Test 3: SE(3) sanity
        orth_err = np.linalg.norm(R @ R.T - np.eye(3), ord='fro')
        det_R = float(np.linalg.det(R))
        if orth_err > se3['worst_orth']:
            se3.update(worst_orth=float(orth_err), cam=cam['name'])
        se3['min_det'] = min(se3['min_det'], det_R)

        # Test 4: K positivity
        fx, fy = float(K[0, 0]), float(K[1, 1])
        if fx < k_pos['fx_min']:
            k_pos.update(fx_min=fx, cam=cam['name'])
        k_pos['fy_min'] = min(k_pos['fy_min'], fy)

    # Gate on each test, raising with a descriptive message.
    if sub_pixel['worst'] >= eps:
        raise ValueError(
            f"ortho->pinhole sub-pixel test failed: worst error "
            f"{sub_pixel['worst']:.4f} px >= {eps} px on cam "
            f"{sub_pixel['cam']} at point {sub_pixel['pt']}")
    if pos_depth['worst'] <= 0:
        raise ValueError(
            f"ortho->pinhole positive-depth test failed: min Z "
            f"{pos_depth['worst']:.4f} <= 0 on cam {pos_depth['cam']} "
            f"at point {pos_depth['pt']}")
    if se3['worst_orth'] >= 1e-5 or se3['min_det'] <= 0.999:
        raise ValueError(
            f"ortho->pinhole SE(3) test failed: worst ||RR^T-I||_F "
            f"{se3['worst_orth']:.2e}, min det(R) {se3['min_det']:.6f} "
            f"(cam {se3['cam']})")
    if k_pos['fx_min'] <= 0 or k_pos['fy_min'] <= 0:
        raise ValueError(
            f"ortho->pinhole K-positivity test failed: min fx "
            f"{k_pos['fx_min']:.4f}, min fy {k_pos['fy_min']:.4f} "
            f"(cam {k_pos['cam']})")

    return {
        'sub_pixel': sub_pixel,
        'positive_depth': pos_depth,
        'se3': se3,
        'k_positivity': k_pos,
    }


def build_pinhole_cameras(calib_dir, recording_dir, gt_xyz,
                          scale=DLT_AB_SCALE, eps=EPS_PIXEL, verbose=True):
    """Build pinhole (K, ext, dist) for every ortho camera in a recording.

    Runs preprocess.md Steps 1-3 + validation tests against the supplied GT
    keypoints ``gt_xyz`` (N, 3). Returns five dicts keyed by camera name:
    ``(intrinsics, extrinsics, distortions, dlt_coefficients, sizes)`` where
    ``sizes[cam] = (W, H)`` and ``distortions[cam] = [0, 0, 0, 0, 0]``.
    """
    cams = load_dlt_cameras(calib_dir, recording_dir, scale=scale)

    X = np.asarray(gt_xyz, dtype=np.float64).reshape(-1, 3)
    X = X[np.isfinite(X).all(axis=1)]
    if X.shape[0] == 0:
        raise ValueError('build_pinhole_cameras: no finite GT keypoints supplied')
    X, n_dropped, _ = robust_filter(X)

    for cam in cams:
        cam['pd'] = orient_proj_dir(cam['A'], X)
    D_star, dinfo = fit_D_star(cams, X, eps=eps)
    results = run_validations(cams, X, D_star, eps=eps)

    if verbose:
        print(f'  ortho->pinhole: {len(cams)} cams, {X.shape[0]} GT pts '
              f'({n_dropped} dropped), D* = {D_star:.4g}')
        print(f'    sub-pixel worst = {results["sub_pixel"]["worst"]:.4f} px, '
              f'min Z = {results["positive_depth"]["worst"]:.3g}, '
              f'min fx = {results["k_positivity"]["fx_min"]:.3g}')

    intrinsics, extrinsics, distortions, dlt_coefficients, sizes = {}, {}, {}, {}, {}
    for cam in cams:
        name = cam['name']
        intrinsics[name] = cam['K'].tolist()
        extrinsics[name] = cam['ext'].tolist()
        distortions[name] = [0.0, 0.0, 0.0, 0.0, 0.0]
        dlt_coefficients[name] = cam['L'].tolist()
        sizes[name] = (cam['W'], cam['H'])

    return intrinsics, extrinsics, distortions, dlt_coefficients, sizes
