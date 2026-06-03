# Ortho camera preprocessing spec

This document specifies the closed-form construction that `../posetail-preprocessing`
should implement to **rewrite each orthographic recording's metadata YAML in
pinhole shape**. The main `posetail` repo then consumes the YAML through the
existing pinhole loader with zero ortho-specific code paths.

## Goal

For each recording with one or more orthographic cameras (typical fly setup:
7 cameras in a multi-view rig sharing GT 3D keypoints), produce a metadata YAML
whose camera fields exactly match the pinhole convention the main repo already
reads (`intrinsic_matrices`, `extrinsic_matrices`, `distortion_matrices` as
top-level dicts keyed by camera name). The (K, ext) pair is a "telephoto
pinhole" that approximates the ortho DLT to within 0.5 pixel on the recording's
GT keypoints; `dist = 0` because ortho cameras have no lens distortion to
model.

```yaml
intrinsic_matrices:                  # NEW — same shape as pinhole
  <cam_name>: [[fx, 0,  cx],
               [0,  fy, cy],
               [0,  0,  1 ]]
  ...
extrinsic_matrices:                  # NEW — same shape as pinhole
  <cam_name>: [[r00, r01, r02, t0],
               [r10, r11, r12, t1],
               [r20, r21, r22, t2],
               [0,   0,   0,   1 ]]   # 4×4 SE(3) world-to-cam
  ...
distortion_matrices:                 # NEW — zeros for ortho
  <cam_name>: [0, 0, 0, 0, 0]
  ...
camera_heights:
  <cam_name>: H                       # existing
  ...
camera_widths:
  <cam_name>: W                       # existing
  ...
offset_dict:                          # existing, optional
  <cam_name>: [ox, oy]
  ...
dlt_coefficients:                    # OPTIONAL — retained for traceability;
  <cam_name>: [11 floats]            #            main repo ignores it
  ...
```

The main repo derives `center = −R^T · t` from `extrinsic_matrices[<cam_name>]`
at load time — the same path it already uses for pinhole. No `cam_type` field is
required; presence of `intrinsic_matrices` is the pinhole signal.

## Inputs to the preprocessing routine

- Existing camera metadata: `L`, `size`, `offset` per ortho camera. The 11-coef
  DLT defines the projection matrix
  ```
  proj_mat = [[L[0]/s, L[1]/s, L[2]/s, L[3]],
              [L[4]/s, L[5]/s, L[6]/s, L[7]],
              [0,      0,      0,      1   ]]
  ```
  with `A = proj_mat[:2, :3]` and `t_proj = proj_mat[:2, 3]`. The forward
  projection is `p2d = A · X + t_proj − offset`.

  **DLT units gotcha (`s` factor).** Different rigs store the multipliers
  `L1..L3, L5..L7` and the GT keypoints `X` in different units. The JARVIS
  fly pipeline divides those six multipliers by `s = 10` while leaving the
  affine constants `L4, L8` alone — i.e. the calibrated DLT is in 10·units
  while the GT CSV is in units (mm). Concretely, projection only matches
  the actual videos after this rescale. The preprocessor must apply the
  per-rig `s` (default 1.0; 10.0 for the johnson lab fly recordings) before
  reading `A` and `t_proj`. A good cross-check is to project a few high-conf
  GT points and confirm they land on the animal in the video.

- A representative set of finite 3D GT keypoints `X ∈ ℝ³` spanning all frames
  and tracked subjects of the recording. Should cover the spatial extent the
  model will see at training time. Use a **confidence threshold** (default
  `> 0.7`) when GT comes from a predictor — low-confidence rows include
  failed triangulations that are placed orders of magnitude outside the
  scene.

## Construction (closed-form, single pass)

### Step 1 — Orient each camera's proj_dir from the data

For each ortho camera `i`:
```
A_i      = proj_mat[:2, :3]                  # 2×3 image-plane projection (from L)
t_proj_i = proj_mat[:2, 3]
raw_pd_i = cross(A_i[0], A_i[1])
pd_i     = raw_pd_i / ‖raw_pd_i‖
if median over X of (X · pd_i) < 0:
    pd_i ← −pd_i                              # data-driven orient
```

### Step 2 — Pick the common rig distance D* (sub-pixel honest)

The constructed pinhole at distance `D` along `−pd` from the lifting plane has
projection deviation from the exact ortho DLT bounded by

```
|Δp2d|(X)  ≈  σ_max(A) · ‖origins(X)‖ · |X·pd| / D    for D ≫ |X·pd|
```

where `origins(X) = A_pinv · (A · X)` is the foot of X on the lifting plane and
`A_pinv = A^T · (A · A^T)⁻¹`. Constrain the worst case across all cameras and
all GT keypoints to `ε = 0.5 px`:

```
For each cam i and each finite GT X:
    origins_i(X) = A_pinv_i · (A_i · X)
    s_i(X)       = X · pd_i

D_pinhole_i     = (2 / ε) · σ_max(A_i) · max_X(‖origins_i(X)‖ · |s_i(X)|)
D_positivity_i  = −min_X (X · pd_i)
D*              = max(  max_i D_pinhole_i,
                        max_i D_positivity_i + small_margin )
```

`small_margin` ≈ 0.1 · scene_extent_along_pd is fine; it just keeps depths
strictly bounded away from zero.

`D*` is a single common value used across all ortho cameras in the recording
(rig-radius interpretation). For typical fly geometry (`σ_max(A) ≈ 50 px/mm`,
scene ~5 mm × 5 mm), `D* ≈ 2.5 m`. Sanity-check the resulting magnitude before
writing out.

### Step 3 — Build (R, t, K, dist, ext) per camera

The camera basis is anchored to `A_i[1]` (the v-axis row of the DLT), with the
u-axis recovered as `u_y × u_z` so that `K[1, 0] = 0`. Any non-orthogonality
between `A_i[0]` and `A_i[1]` is absorbed into the standard upper-triangular
skew slot `K[0, 1]`. The optical axis sign `±pd` is then chosen per camera so
that `fx > 0` *and* `det(R) = +1`: cameras whose DLT is "mirror-imaged"
relative to the naive `u_z = pd` choice are handled by flipping `u_z = −pd`
and placing the camera centre on the `+pd` side. Both branches yield proper
SE(3) extrinsics with positive `K` diagonals — downstream Rodrigues /
quaternion paths in the main repo work unchanged.

For each ortho camera:
```
u_y          = A_i[1] / ‖A_i[1]‖                 # A[1] ⊥ pd by construction
u_x_proper   = u_y × pd_i                        # candidate if u_z = +pd
if A_i[0] · u_x_proper >= 0:
    u_z = +pd_i
    u_x = u_x_proper
    C   = −D* · pd_i                             # camera centre at −D*·pd
else:
    u_z = −pd_i
    u_x = −u_x_proper                            # = u_y × u_z; keeps right-handed basis
    C   = +D* · pd_i                             # camera centre at +D*·pd

R   = [u_x ; u_y ; u_z]                          # 3×3 world-to-cam, det(R) = +1
t   = −R · C                                     # t[2] = +D* in both branches
ext = [[R, t.reshape(3,1)],
       [0, 0, 0, 1]]                             # 4×4 SE(3) world-to-cam

fx   = D* · (A_i[0] · u_x)                       # > 0 by construction
fy   = D* · ‖A_i[1]‖                             # > 0 always
skew = D* · (A_i[0] · u_y)                       # K[0, 1]; zero iff A[0] ⊥ A[1]
cx   = t_proj_i[0]                               # principal point x
cy   = t_proj_i[1]                               # principal point y
K    = [[fx, skew, cx],
        [0,  fy,   cy],
        [0,  0,    1 ]]                          # 3×3 intrinsic matrix
dist = [0, 0, 0, 0, 0]                           # ortho has no lens distortion
```

Camera-frame depth at `X` is `(R · X + t)[2] = u_z · X + D*`, which is
`D* + pd · X` for the `u_z = +pd` branch and `D* − pd · X` for the
`u_z = −pd` branch. Both lie in `(0, 2·D*)` for every finite GT — positive
by construction.

Write each `K` (3×3 float, row-major) into the YAML under the top-level
`intrinsic_matrices` dict; each `ext` under `extrinsic_matrices`; and
`distortion_matrices[<cam_name>] = [0, 0, 0, 0, 0]`. All three dicts match the
existing pinhole convention exactly.

### Step 4 (optional ergonomics) — pre-translate the world frame

If `mean(GT)` is far from the world origin, the magnitude of `D*` and the
numerical conditioning of downstream training improve when the scene is
recentred. The preprocessor may:
1. Compute `c = mean(GT)`.
2. Translate GT by `−c`, and update each cam's `t_proj_i ← t_proj_i + A_i · c`.
3. Run Steps 1-3 on the translated frame.
4. Translate the resulting camera centres back to the original world frame by
   adding `c` to each `C` before recomputing `t = −R · C`.

This is purely optional; the math is correct without it.

## Validation tests the preprocessor should run before writing YAMLs

Each test should abort the preprocessing run with a clear error identifying the
camera and the failing keypoint when it fails.

Before running the tests below, **robust-filter the GT keypoints**: real
prediction CSVs include failed-triangulation outliers far outside the scene.
Drop any point whose `(x, y, z)` is more than `K · MAD` from the per-coord
median (default `K = 10`). Without this filter, a single outlier dominates the
worst-case in test 1 and inflates `D*` by orders of magnitude.

1. **Sub-pixel honesty.** For every cam `i` and every (filtered) GT `X`:
   ```
   ‖p2d_pinhole(X; K_i, ext_i) − p2d_DLT(X; L_i)‖ < 0.5 px
   ```
   where `p2d_pinhole` is the standard pinhole projection `K · (R·X + t)` with
   homogeneous divide.
2. **Positive depth.** For every cam `i` and every (filtered) GT `X`:
   ```
   (R_i · X + t_i)[2] > 0
   ```
   i.e., GT is in front of the camera in the constructed pinhole frame.
   Equivalently, `(X − center_i) · pd_i > 0` where `center_i = −R_i^T · t_i`.
3. **SE(3) sanity.** `‖R · R^T − I‖_F < 1e-5` and `det(R) > 0.999`.
4. **K positivity.** `fx > 0` and `fy > 0`.

Informational (report but don't gate):
- `|A[0] · A[1]| / (‖A[0]‖ · ‖A[1]‖)` per camera — the pixel-axis skew that
  ends up as `K[0, 1]`. Real DLT calibrations typically show 1e−3 to 1e−2.
- Count of cameras taking the `u_z = −pd` branch (cf. Step 3).

## Out of scope for the preprocessor

- Do NOT store `center` — it's derived from `extrinsic_matrices[<cam>]` at
  load time (the main repo uses the same `center = −R^T · t` derivation it
  already uses for pinhole cameras).
- Pinhole cameras: their YAML schema is unchanged.
- Recordings where ortho and pinhole cameras coexist in one rig: irrelevant —
  after this preprocessor runs, ortho recordings are byte-equivalent to
  pinhole recordings, so the main repo's homogeneous-group assumption holds
  trivially.
