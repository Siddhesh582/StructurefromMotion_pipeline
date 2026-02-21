import numpy as np
import gtsam
from gtsam import symbol_shorthand

X = symbol_shorthand.X   # camera poses
L = symbol_shorthand.L   # 3D landmarks


#  Internal helper

def _to_dict(camera_transforms):
    """Normalise a list or dict of transforms to a dict keyed by cam_idx."""
    if isinstance(camera_transforms, list):
        return {i: v for i, v in enumerate(camera_transforms)}
    return camera_transforms


#  Reprojection error

def compute_reprojection_errors(observations, points_3d, camera_transforms, K):
    """
    Compute per-observation reprojection errors.

    Args:
        observations      : list of (cam_idx, pt_idx, 2d_pt)
        points_3d         : (M, 3) world-frame 3D points
        camera_transforms : dict {cam_idx: {'R','t'}} or list
        K                 : (3, 3) intrinsic matrix

    Returns:
        errors : (N,) array of pixel errors
    """
    tf_dict = _to_dict(camera_transforms)
    errors  = []

    for cam_idx, pt_idx, meas in observations:
        if pt_idx >= len(points_3d) or cam_idx not in tf_dict:
            continue
        X_w = points_3d[pt_idx].reshape(3, 1)
        if np.any(np.isnan(X_w)) or np.any(np.isinf(X_w)):
            continue
        tf    = tf_dict[cam_idx]
        R     = tf['R']
        t     = tf['t'].reshape(3, 1)
        X_cam = R @ X_w + t
        if X_cam[2, 0] <= 1e-3:
            continue
        proj  = K @ X_cam
        pixel = (proj[:2] / proj[2]).flatten()
        if np.any(np.isnan(pixel)) or np.any(np.isinf(pixel)):
            continue
        err = np.linalg.norm(pixel - meas)
        if np.isfinite(err):
            errors.append(err)

    return np.array(errors)


def filter_observations(observations, points_3d, camera_transforms,
                        K, max_reproj_error=10.0):
    """
    Remove observations with reprojection error above max_reproj_error.
    Returns filtered observation list.
    """
    tf_dict = _to_dict(camera_transforms)
    clean   = []

    for cam_idx, pt_idx, meas in observations:
        if pt_idx >= len(points_3d) or cam_idx not in tf_dict:
            continue
        X_w = points_3d[pt_idx].reshape(3, 1)
        if np.any(np.isnan(X_w)) or np.any(np.isinf(X_w)):
            continue
        tf    = tf_dict[cam_idx]
        X_cam = tf['R'] @ X_w + tf['t'].reshape(3, 1)
        if X_cam[2, 0] <= 0:
            continue
        proj  = K @ X_cam
        pixel = (proj[:2] / proj[2]).flatten()
        if np.linalg.norm(pixel - meas) <= max_reproj_error:
            clean.append((cam_idx, pt_idx, meas))

    return clean


def print_reprojection_stats(errors, label=''):
    tag = f' ({label})' if label else ''
    print(f"\nReprojection errors{tag}:")
    if len(errors) == 0:
        print("  WARNING: no valid observations to compute errors")
        return
    print(f"  Mean   : {np.mean(errors):.4f} px")
    print(f"  Median : {np.median(errors):.4f} px")
    print(f"  Std    : {np.std(errors):.4f} px")
    print(f"  Max    : {np.max(errors):.4f} px")


#  Observation map (for PnP-based pipeline)

def create_observation_map(point_tracks, camera_transforms, points_3d):
    """
    Build a flat observation list from a point_tracks dict.
    camera_transforms can be a dict or list.
    """
    tf_dict = _to_dict(camera_transforms)

    track_cam_indices = {c for v in point_tracks.values() for c in v.keys()}
    print(f"  point_tracks entries     : {len(point_tracks)}")
    print(f"  point_tracks cam indices : {sorted(track_cam_indices)}")
    print(f"  camera_transforms keys   : {sorted(tf_dict.keys())}")
    overlap = track_cam_indices & set(tf_dict.keys())
    print(f"  overlapping keys         : {sorted(overlap)}")

    observations     = []
    cameras_with_obs = set()

    for pt_idx, cam_obs in point_tracks.items():
        if pt_idx >= len(points_3d):
            continue
        for cam_idx, pt_2d in cam_obs.items():
            if cam_idx not in tf_dict:
                continue
            observations.append((cam_idx, pt_idx, pt_2d))
            cameras_with_obs.add(cam_idx)

    obs_per_cam   = {}
    obs_per_point = {}
    for cam_idx, pt_idx, _ in observations:
        obs_per_cam[cam_idx]  = obs_per_cam.get(cam_idx, 0)  + 1
        obs_per_point[pt_idx] = obs_per_point.get(pt_idx, 0) + 1

    obs_counts = list(obs_per_point.values())
    print(f"\nObservation map:")
    print(f"  Total 3D points         : {len(points_3d)}")
    print(f"  Points with observations: {len(obs_per_point)}")
    print(f"  Total observations      : {len(observations)}")
    print(f"  Cameras with obs        : {len(cameras_with_obs)}")
    print(f"  Avg obs / point         : {len(observations)/max(len(obs_per_point),1):.2f}")
    print(f"  Avg obs / camera        : {len(observations)/max(len(cameras_with_obs),1):.2f}")
    if obs_counts:
        print(f"  Obs/point — min:{min(obs_counts)}  "
              f"max:{max(obs_counts)}  median:{np.median(obs_counts):.1f}")
    else:
        print("  WARNING: no observations found — check point_tracks and cam_transforms")

    return observations, cameras_with_obs


#  Bundle adjustment

def bundle_adjustment_gtsam(observations, points_3d_initial,
                             camera_transforms, K,
                             cameras_with_obs,
                             num_iterations=100,
                             pixel_sigma=1.5,
                             verbose=True):
    """
    Full bundle adjustment using GTSAM Levenberg-Marquardt.

    Args:
        observations       : list of (cam_idx, pt_idx, 2d_pt)
        points_3d_initial  : (M, 3) initial world-frame 3D points
        camera_transforms  : dict {cam_idx: {'R','t'}} or list
        K                  : (3, 3) intrinsic matrix
        cameras_with_obs   : set of cam_idx with valid observations
        num_iterations     : max LM iterations
        pixel_sigma        : measurement noise sigma (pixels)
        verbose            : print LM summary

    Returns:
        optimized_points : (M, 3)
        optimized_poses  : dict {cam_idx: {'R','t'}}
        result           : raw gtsam.Values
    """
    print("\n" + "="*80)
    print("BUNDLE ADJUSTMENT — GTSAM")

    tf_dict = _to_dict(camera_transforms)
    graph   = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    cal    = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)

    meas_noise  = gtsam.noiseModel.Isotropic.Sigma(2, pixel_sigma)
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.01, 0.01, 0.01, 0.001, 0.001, 0.001])
    )

    # Camera poses
    cams_sorted   = sorted(cameras_with_obs)
    inserted_cams = set()
    print(f"\nAdding {len(cams_sorted)} camera poses...")

    for cam_idx in cams_sorted:
        if cam_idx not in tf_dict:
            print(f"  WARNING: cam {cam_idx} not in tf_dict — skipping")
            continue
        tf    = tf_dict[cam_idx]
        R_w2c = tf['R']
        t_w2c = tf['t'].reshape(3, 1)
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c
        pose  = gtsam.Pose3(gtsam.Rot3(R_c2w),
                            gtsam.Point3(t_c2w.flatten()))
        initial.insert(X(cam_idx), pose)
        inserted_cams.add(cam_idx)

        if cam_idx == cams_sorted[0]:
            graph.add(gtsam.PriorFactorPose3(X(cam_idx), pose, prior_noise))
            print(f"  Anchored camera {cam_idx}")

    # 3D points
    pts_with_obs = sorted({pt_idx for _, pt_idx, _ in observations
                           if pt_idx < len(points_3d_initial)})
    pts_set      = set(pts_with_obs)
    print(f"Adding {len(pts_with_obs)} 3D points...")

    for pt_idx in pts_with_obs:
        initial.insert(L(pt_idx), gtsam.Point3(points_3d_initial[pt_idx]))

    # Projection factors
    n_factors = 0
    for cam_idx, pt_idx, meas in observations:
        if cam_idx not in inserted_cams or pt_idx not in pts_set:
            continue
        graph.add(gtsam.GenericProjectionFactorCal3_S2(
            gtsam.Point2(float(meas[0]), float(meas[1])),
            meas_noise,
            X(cam_idx), L(pt_idx), cal
        ))
        n_factors += 1
    print(f"Added {n_factors} projection factors")

    # Optimize
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY" if verbose else "SILENT")
    params.setMaxIterations(num_iterations)
    params.setRelativeErrorTol(1e-5)
    params.setAbsoluteErrorTol(1e-5)

    optimizer   = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    init_error  = graph.error(initial)
    print(f"\nInitial graph error : {init_error:.6f}")

    result      = optimizer.optimize()
    final_error = graph.error(result)
    print(f"Final graph error   : {final_error:.6f}")

    if init_error > 1e-10:
        print(f"Error reduction     : {(1 - final_error/init_error)*100:.2f}%")
    else:
        print("  WARNING: initial error is zero — graph has no factors")
    print(f"Iterations used     : {optimizer.iterations()}")

    # Diagnostics
    _print_pose_diagnostics(tf_dict, cams_sorted, result)
    _print_point_diagnostics(points_3d_initial, pts_with_obs, result)

    # Extract results
    optimized_poses  = _extract_poses(tf_dict, cameras_with_obs, result)
    optimized_points = _extract_points(points_3d_initial, pts_with_obs,
                                        pts_set, result)

    print(f"\nOptimized {len(cams_sorted)} cameras and {len(pts_with_obs)} points")
    return optimized_points, optimized_poses, result


##

def _print_pose_diagnostics(tf_dict, cams_sorted, result):
    max_rot, max_trans = 0.0, 0.0
    for cam_idx in cams_sorted:
        if cam_idx not in tf_dict:
            continue
        R_orig = tf_dict[cam_idx]['R']
        t_orig = tf_dict[cam_idx]['t'].flatten()
        pose   = result.atPose3(X(cam_idx))
        R_c2w  = pose.rotation().matrix()
        t_c2w  = pose.translation()
        R_opt  = R_c2w.T
        t_opt  = (-R_opt @ t_c2w.reshape(3, 1)).flatten()
        max_rot   = max(max_rot,   np.linalg.norm(R_orig - R_opt, 'fro'))
        max_trans = max(max_trans, np.linalg.norm(t_orig - t_opt))
    print(f"\nCamera changes — max rot (Frobenius): {max_rot:.6f}  "
          f"max trans: {max_trans:.6f}")


def _print_point_diagnostics(points_3d_initial, pts_with_obs, result):
    max_shift = 0.0
    for pt_idx in pts_with_obs:
        if pt_idx >= len(points_3d_initial):
            continue
        p_opt     = result.atPoint3(L(pt_idx))
        p_arr     = p_opt.flatten() if isinstance(p_opt, np.ndarray) \
                    else np.array([p_opt.x(), p_opt.y(), p_opt.z()])
        max_shift = max(max_shift,
                        np.linalg.norm(points_3d_initial[pt_idx] - p_arr))
    print(f"Point changes  — max shift: {max_shift:.6f}")


def _extract_poses(tf_dict, cameras_with_obs, result):
    optimized = {}
    for cam_idx, tf in tf_dict.items():
        if cam_idx in cameras_with_obs:
            pose  = result.atPose3(X(cam_idx))
            R_c2w = pose.rotation().matrix()
            t_c2w = pose.translation()
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w.reshape(3, 1)
            optimized[cam_idx] = {'R': R_w2c, 't': t_w2c}
        else:
            optimized[cam_idx] = tf
    return optimized


def _extract_points(points_3d_initial, pts_with_obs, pts_set, result):
    optimized = np.zeros_like(points_3d_initial)
    for pt_idx in range(len(points_3d_initial)):
        if pt_idx in pts_set:
            p = result.atPoint3(L(pt_idx))
            optimized[pt_idx] = p.flatten() if isinstance(p, np.ndarray) \
                                 else np.array([p.x(), p.y(), p.z()])
        else:
            optimized[pt_idx] = points_3d_initial[pt_idx]
    return optimized