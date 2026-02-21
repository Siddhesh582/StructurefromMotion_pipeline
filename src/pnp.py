import cv2
import numpy as np

from src.pointcloud import PointCloud


def match_image_pair(img1, img2, n_features=5000, ratio=0.75):
    sift  = cv2.SIFT_create(nfeatures=n_features)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return None, None

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw     = matcher.knnMatch(des1, des2, k=2)
    good    = [m for m, n in raw if m.distance < ratio * n.distance]

    if len(good) < 8:
        return None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
    return src_pts, dst_pts


def compute_essential_matrix(src_pts, dst_pts, K):
    E, mask = cv2.findEssentialMat(
        src_pts, dst_pts, K, cv2.RANSAC, 0.999, 1.0
    )
    inlier_src = src_pts[mask.ravel() == 1]
    inlier_dst = dst_pts[mask.ravel() == 1]
    return E, inlier_src, inlier_dst


def recover_pose_from_E(E, src_pts, dst_pts, K):
    """
    SVD decomposition of E to recover R and t.
    Selects the solution with the most points in front of both cameras.
    """
    U, _, Vt = np.linalg.svd(E)
    Y = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R1 = U @ Y   @ Vt
    R2 = U @ Y.T @ Vt
    t1 =  U[:, 2]
    t2 = -U[:, 2]

    candidates = [(R1, t1), (R1, t2), (R2, t1), (R2, t2)]
    candidates = [(-R, -t) if np.linalg.det(R) < 0 else (R, t)
                  for R, t in candidates]

    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])

    best_count, best_R, best_t = -1, None, None
    for R, t in candidates:
        P2    = K @ np.hstack([R, t.reshape(3, 1)])
        count = 0
        for p1, p2 in zip(src_pts, dst_pts):
            X    = cv2.triangulatePoints(P1, P2,
                                          p1.reshape(2, 1),
                                          p2.reshape(2, 1))
            X   /= X[3]
            pt3  = X[:3].flatten()
            z1   = pt3[2]
            z2   = float(R[2, :] @ (pt3 - t))
            if z1 > 0 and z2 > 0:
                count += 1
        if count > best_count:
            best_count, best_R, best_t = count, R, t

    return best_R, best_t


def triangulate_from_poses(P1, P2, src_pts, dst_pts):
    """
    Triangulate 2D correspondences given two projection matrices.
    Keeps only points with positive depth.
    """
    pts4d  = cv2.triangulatePoints(P1, P2, src_pts.T, dst_pts.T)
    pts4d /= pts4d[3, :]
    pts3d  = pts4d[:3, :].T

    mask   = pts3d[:, 2] > 0
    return pts3d[mask], src_pts[mask], dst_pts[mask]


def estimate_pose_pnp(pts_3d, pts_2d, K):
    """
    Estimate camera pose via PnP RANSAC.
    Returns 4x4 world-to-cam transform or None on failure.
    """
    success, rvec, t, inliers = cv2.solvePnPRansac(
        pts_3d.astype(np.float32),
        pts_2d.astype(np.float32),
        K, None
    )
    if not success:
        return None

    R = cv2.Rodrigues(rvec)[0]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t.flatten()
    return T


def incremental_sfm(images, K,
                    n_features_first=1000,
                    n_features_rest=5000,
                    ratio=0.75,
                    filter_percentile=90):
    """
    Forward incremental SfM:
      - Frame 0--> 1 : recover pose from E, triangulate seed points
      - Frame i--> i+1 : PnP from existing cloud, triangulate new points
    """
    pc = PointCloud()
    pc.camera_poses.append(np.eye(4))   # camera 0 at world origin
    n  = len(images)

    for i in range(n - 1):
        j  = i + 1
        nf = n_features_first if i == 0 else n_features_rest
        print(f"\n{'='*50}")
        print(f"Frame {i} -->  {j}")

        src_pts, dst_pts = match_image_pair(images[i], images[j],
                                             n_features=nf, ratio=ratio)
        if src_pts is None:
            print("  SKIP: insufficient matches")
            pc.camera_poses.append(pc.camera_poses[-1])
            continue

        print(f"  Matches: {len(src_pts)}")

        E, src_pts, dst_pts = compute_essential_matrix(src_pts, dst_pts, K)
        print(f"  E inliers: {len(src_pts)}")

        if i == 0:
            # Seed pair — recover pose from E and triangulate
            R, t  = recover_pose_from_E(E, src_pts, dst_pts, K)
            T1    = np.eye(4)
            T1[:3, :3] = R
            T1[:3,  3] = t
            pc.camera_poses.append(T1)

            P1    = K @ pc.camera_poses[0][:3, :]
            P2    = K @ pc.camera_poses[1][:3, :]
            pts3d, src_f, dst_f = triangulate_from_poses(P1, P2,
                                                          src_pts, dst_pts)
            pc.add_points(pts3d, src_f, dst_f, 0, 1)
            print(f"  Seed: {len(pts3d)} points")

        else:
            # Subsequent frames — PnP then triangulate new points
            common_3d, common_2d = pc.common_pts(src_pts, dst_pts)
            print(f"  Common 3D-2D: {len(common_3d)}")

            if len(common_3d) < 6:
                print("  Too few common pts — skipping frame")
                pc.camera_poses.append(pc.camera_poses[-1])
                continue

            T_new = estimate_pose_pnp(common_3d, common_2d, K)
            if T_new is None:
                print("  PnP failed — skipping frame")
                pc.camera_poses.append(pc.camera_poses[-1])
                continue

            pc.camera_poses.append(T_new)
            print("  PnP SUCCESS")

            P1    = K @ pc.camera_poses[i][:3, :]
            P2    = K @ pc.camera_poses[j][:3, :]
            pts3d, src_f, dst_f = triangulate_from_poses(P1, P2,
                                                          src_pts, dst_pts)
            pc.add_points(pts3d, src_f, dst_f, i, j)
            print(f"  +{len(pts3d)} pts  |  cloud: {len(pc)}")

    print(f"\nFiltering outliers (percentile={filter_percentile})...")
    pc.filter_points(filter_percentile)
    print(f"Final: {len(pc)} points, {len(pc.camera_poses)} cameras")
    return pc