import numpy as np


#  Point Normalization

def normalize_points(pts):
    """Isotropic normalization: zero-mean, avg distance = sqrt(2)."""
    centroid  = np.mean(pts, axis=0)
    pts_c     = pts - centroid
    avg_dist  = np.mean(np.sqrt(np.sum(pts_c**2, axis=1)))

    if avg_dist < 1e-10:
        raise ValueError("Points are nearly identical — cannot normalize.")

    scale = np.sqrt(2) / avg_dist
    T = np.array([
        [scale, 0,     -scale * centroid[0]],
        [0,     scale, -scale * centroid[1]],
        [0,     0,      1]
    ])
    pts_h      = np.hstack([pts, np.ones((len(pts), 1))])
    pts_norm_h = (T @ pts_h.T).T
    pts_norm   = pts_norm_h[:, :2] / pts_norm_h[:, 2:]
    return pts_norm, T


#  8-Point Algorithm

def F_matrix(pts1, pts2):
    """Normalized 8-point algorithm for the Fundamental matrix."""
    if len(pts1) < 8:
        raise ValueError("Need at least 8 point correspondences.")

    pts1_n, T1 = normalize_points(pts1)
    pts2_n, T2 = normalize_points(pts2)

    n = pts1_n.shape[0]
    A = np.zeros((n, 9))
    for i in range(n):
        x1, y1 = pts1_n[i]
        x2, y2 = pts2_n[i]
        A[i] = [x2*x1, x2*y1, x2,
                y2*x1, y2*y1, y2,
                x1,    y1,    1]

    _, _, Vt = np.linalg.svd(A)
    F_norm   = Vt[-1].reshape(3, 3)

    # Denormalize
    F = T2.T @ F_norm @ T1

    # Enforce rank-2
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vt
    return F


#  Error Metrics

def epipolar_constraint_error(F, pts1, pts2):
    """Symmetric epipolar (point-to-line) distance."""
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
    pts2_h = np.hstack([pts2, np.ones((len(pts2), 1))])

    lines2 = (F   @ pts1_h.T).T   # epipolar lines in image 2
    lines1 = (F.T @ pts2_h.T).T   # epipolar lines in image 1

    def point_line_dist(pts_h, lines):
        num   = np.abs(np.sum(pts_h * lines, axis=1))
        denom = np.sqrt(lines[:, 0]**2 + lines[:, 1]**2)
        return num / np.maximum(denom, 1e-10)

    d1 = point_line_dist(pts1_h, lines1)
    d2 = point_line_dist(pts2_h, lines2)
    return (d1 + d2) / 2.0


def sampson_distance(F, pts1, pts2):
    """
    Sampson distance — first-order approximation to reprojection error.
    More accurate than epipolar line distance for RANSAC scoring.
    """
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
    pts2_h = np.hstack([pts2, np.ones((len(pts2), 1))])

    Fp1  = (F   @ pts1_h.T).T
    Ftp2 = (F.T @ pts2_h.T).T

    num   = np.sum(pts2_h * Fp1, axis=1) ** 2
    denom = Fp1[:,0]**2 + Fp1[:,1]**2 + Ftp2[:,0]**2 + Ftp2[:,1]**2
    return num / np.maximum(denom, 1e-10)


#  RANSAC

def _adaptive_iterations(inlier_ratio, confidence=0.999, sample_size=8, max_iter=10000):
    if inlier_ratio <= 0 or inlier_ratio >= 1:
        return max_iter
    denom = 1 - inlier_ratio**sample_size
    if denom < 1e-10:
        return 1
    log_denom = np.log(denom)
    if log_denom >= 0 or not np.isfinite(log_denom):
        return 1
    n = int(np.log(1 - confidence) / log_denom)
    return max(1, min(n, max_iter))


def ransac_fundamental_matrix(pts1, pts2,
                               num_iterations=2000,
                               threshold=1.5,
                               confidence=0.999):
    """
    RANSAC to robustly estimate F.
    Uses Sampson distance as the inlier metric.
    Iteration count adapts once a reasonable inlier ratio is found.
    """
    n = pts1.shape[0]
    best_F, best_inliers, best_n_inliers = None, None, 0
    max_iter = num_iterations

    it = 0
    while it < max_iter:
        # Sample 8 points
        idx = np.random.choice(n, 8, replace=False)
        try:
            F = F_matrix(pts1[idx], pts2[idx])
        except (np.linalg.LinAlgError, ValueError):
            it += 1
            continue

        errors  = sampson_distance(F, pts1, pts2)
        inliers = errors < threshold
        n_inl   = inliers.sum()

        if n_inl > best_n_inliers:
            best_F, best_inliers, best_n_inliers = F, inliers, n_inl

            # Adaptive update
            ratio    = best_n_inliers / n
            max_iter = min(max_iter,
                           _adaptive_iterations(ratio, confidence))

            # Early exit if nearly all points are inliers
            if ratio > 0.95:
                break

        it += 1

    # ── Refinement on inliers ──
    in_pts1 = pts1[best_inliers]
    in_pts2 = pts2[best_inliers]
    if len(in_pts1) >= 8:
        try:
            best_F = F_matrix(in_pts1, in_pts2)
            # Re-evaluate inliers with refined F
            errors       = sampson_distance(best_F, pts1, pts2)
            best_inliers = errors < threshold
            best_n_inliers = best_inliers.sum()
        except (np.linalg.LinAlgError, ValueError):
            pass  # keep the previous best_F

    ratio = best_n_inliers / n
    print(f"  RANSAC done  |  inliers: {best_n_inliers}/{n}  ({ratio:.1%})"
          f"  |  iterations used: {it}")
    return best_F, best_inliers