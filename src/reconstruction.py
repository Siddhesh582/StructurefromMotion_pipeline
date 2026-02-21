import numpy as np


# ─────────────────────────────────────────────
#  Essential Matrix & Pose Candidates
# ─────────────────────────────────────────────

def essential_matrix(F, K):
    """
    Compute E from F and K, enforce the two-equal-singular-values constraint,
    and return the 4 (R, t) candidate poses.
    """
    E = K.T @ F @ K

    U, S, Vt = np.linalg.svd(E)

    # Fix improper rotations
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1

    # Enforce E's singular values = [1, 1, 0]
    S_enforced    = np.array([1.0, 1.0, 0.0])
    E             = U @ np.diag(S_enforced) @ Vt

    W = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ], dtype=np.float64)

    R1 = U @ W   @ Vt
    R2 = U @ W.T @ Vt
    t  = U[:, 2].reshape(3, 1)

    poses = [
        (R1,  t),
        (R1, -t),
        (R2,  t),
        (R2, -t),
    ]
    return E, poses


# ─────────────────────────────────────────────
#  Triangulation
# ─────────────────────────────────────────────

def triangulate_points(pts1, pts2, K, R, t):
    """
    DLT triangulation for N point correspondences.
    Returns (N, 3) array; invalid points are set to NaN.
    """
    if t.ndim == 1:
        t = t.reshape(3, 1)

    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])

    n  = pts1.shape[0]
    X  = np.full((n, 3), np.nan)

    for i in range(n):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A = np.array([
            x1 * P1[2] - P1[0],
            y1 * P1[2] - P1[1],
            x2 * P2[2] - P2[0],
            y2 * P2[2] - P2[1],
        ])
        try:
            _, S, Vt = np.linalg.svd(A)
            X_h = Vt[-1]

            # Condition number guard
            cond = S[0] / S[-1] if S[-1] > 1e-10 else np.inf
            if cond > 1e6:
                continue

            if np.abs(X_h[3]) < 1e-10:
                continue

            X_3d = X_h[:3] / X_h[3]

            # Depth sanity check
            if np.abs(X_3d[2]) > 1000:
                continue

            X[i] = X_3d
        except np.linalg.LinAlgError:
            continue

    return X


def check_cheirality(X, R, t):
    """
    Returns a boolean mask: True where the 3D point is in front of BOTH cameras.
    """
    t = t.reshape(3, 1)
    depth_c1 = X[:, 2]                       # Z in camera 1 frame
    X_c2     = (R @ X.T + t).T
    depth_c2 = X_c2[:, 2]                    # Z in camera 2 frame
    valid    = (depth_c1 > 0) & (depth_c2 > 0) & ~np.isnan(X[:, 0])
    return valid


def reprojection_error(X, pts, P):
    """
    Mean reprojection error for a set of 3D points and their 2D observations.
    X   : (N, 3)
    pts : (N, 2)
    P   : (3, 4) projection matrix
    """
    X_h   = np.hstack([X, np.ones((len(X), 1))])
    proj  = (P @ X_h.T).T
    proj  = proj[:, :2] / proj[:, 2:3]
    errs  = np.linalg.norm(proj - pts, axis=1)
    return errs


# ─────────────────────────────────────────────
#  Pose Selection (cheirality + reprojection)
# ─────────────────────────────────────────────

SOLUTION_LABELS = [
    "R1 = U·W·Vt,   +t",
    "R1 = U·W·Vt,   -t",
    "R2 = U·Wt·Vt,  +t",
    "R2 = U·Wt·Vt,  -t",
]

def select_pose(poses, pts1, pts2, K, reproj_threshold=2.0):
    """
    Pick the (R, t) with the most cheirality-valid points and lowest
    mean reprojection error among those valid points.

    Returns:
        best_R, best_t, best_X, best_mask, best_idx
    """
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])

    best_count  = -1
    best_R      = None
    best_t      = None
    best_X      = None
    best_mask   = None
    best_idx    = -1
    valid_counts = []

    for sol_idx, (R, t) in enumerate(poses):
        X    = triangulate_points(pts1, pts2, K, R, t)
        mask = check_cheirality(X, R, t)
        n_valid = mask.sum()
        valid_counts.append(n_valid)

        if n_valid > best_count:
            # Secondary criterion: mean reprojection error on valid subset
            P2   = K @ np.hstack([R, t.reshape(3, 1)])
            errs = reprojection_error(X[mask], pts1[mask], P1)
            mean_err = errs.mean() if len(errs) > 0 else np.inf

            # Only prefer if reprojection is reasonable
            if mean_err < reproj_threshold * 3:
                best_count = n_valid
                best_R, best_t = R, t.reshape(3, 1)
                best_X    = X
                best_mask = mask
                best_idx  = sol_idx

    return best_R, best_t, best_X, best_mask, best_idx, valid_counts


def process_all_pairs(camera_poses, reproj_threshold=2.0):
    """
    For every pair in camera_poses that has E/poses computed,
    select the best pose and store triangulated points.
    Prints a per-pair and summary table.
    """
    for idx, pose_data in enumerate(camera_poses):
        i, j     = pose_data['image_pair']
        pts1     = pose_data['inlier_pts1']
        pts2     = pose_data['inlier_pts2']
        K        = pose_data['K']
        poses    = pose_data['poses']

        print(f"\nPAIR {idx+1}/{len(camera_poses)}: Image {i+1} ↔ Image {j+1}"
              f"  ({len(pts1)} inlier correspondences)")

        best_R, best_t, best_X, best_mask, best_idx, valid_counts = \
            select_pose(poses, pts1, pts2, K, reproj_threshold)

        for s, vc in enumerate(valid_counts):
            print(f"  Sol {s+1}: {vc} valid pts")

        if best_R is None or best_X is None or best_mask is None:
            print("  WARNING: no valid pose found for this pair — skipping")
            continue

        if best_mask.sum() < 0.5 * len(pts1):
            print(f"  WARNING: only {best_mask.sum()}/{len(pts1)} valid pts — low confidence")

        # Reprojection error on selected solution
        P1        = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        valid_X   = best_X[best_mask]
        valid_pts = pts1[best_mask]
        errs      = reprojection_error(valid_X, valid_pts, P1)

        print(f"\n  Selected: Solution {best_idx+1}  ({SOLUTION_LABELS[best_idx]})")
        print(f"  Valid pts: {best_mask.sum()}/{len(pts1)}")
        print(f"  Reproj error (cam1): mean={errs.mean():.3f}px  "
              f"median={np.median(errs):.3f}px  max={errs.max():.3f}px")
        print(f"  t = [{best_t[0,0]:.5f}, {best_t[1,0]:.5f}, {best_t[2,0]:.5f}]")
        print(f"  det(R)={np.linalg.det(best_R):.6f}  ||t||={np.linalg.norm(best_t):.6f}")

        pose_data['selected_R']            = best_R
        pose_data['selected_t']            = best_t
        pose_data['selected_idx']          = best_idx
        pose_data['num_valid_3d']          = int(best_mask.sum())
        pose_data['triangulated_points']   = best_X
        pose_data['valid_3d_mask']         = best_mask
        pose_data['reproj_errors']         = errs
        # Store inlier mask aligned to all_matches kp_idx arrays
        pose_data['inlier_mask']           = best_mask

    # ── Summary table ──
    print(f"\n{'Pair':<6} {'Images':<12} {'Solution':<20} {'Valid':<14} "
          f"{'Reproj(mean)':<14} {'Translation'}")
    print("─" * 90)
    solution_short = ["Sol1(R1,+t)", "Sol2(R1,-t)", "Sol3(R2,+t)", "Sol4(R2,-t)"]
    for idx, pd in enumerate(camera_poses):
        if 'selected_idx' not in pd:
            continue
        i, j    = pd['image_pair']
        t       = pd['selected_t']
        total   = len(pd['inlier_pts1'])
        valid   = pd['num_valid_3d']
        err     = pd['reproj_errors'].mean()
        sol_lbl = solution_short[pd['selected_idx']]
        print(f"{idx+1:<6} {f'{i+1}↔{j+1}':<12} {sol_lbl:<20} {f'{valid}/{total}':<14}"
              f"{err:<14.3f} [{t[0,0]:7.4f}, {t[1,0]:7.4f}, {t[2,0]:7.4f}]")

    return camera_poses