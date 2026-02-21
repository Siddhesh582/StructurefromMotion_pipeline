import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gtsam
from gtsam import symbol_shorthand
import matplotlib
matplotlib.use('Agg')

from src.io                import load_image_paths, read_images, get_K_from_colmap
from src.features          import image_features
from src.matching          import build_matches
from src.geometry          import ransac_fundamental_matrix
from src.reconstruction    import essential_matrix, process_all_pairs
from src.cam_trajectory    import compute_camera_poses, plot_camera_trajectory
from src.cam_traj_eval     import compare_trajectories, compute_baselines
from src.pointcloud        import PointCloud
from src.pnp               import incremental_sfm
from src.visualization     import plot_camera_trajectory_labeled, visualize_reconstruction_plotly
from src.bundle_adjustment import (compute_reprojection_errors,
                                    print_reprojection_stats,
                                    filter_observations,
                                    create_observation_map,
                                    bundle_adjustment_gtsam)

# ── 1. Load images ───────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image_dir   = os.path.join(ROOT, 'data', 'images', 'buddha_images')
colmap_path = os.path.join(ROOT, 'data', 'colmap', 'cameras.bin')
images_bin  = os.path.join(ROOT, 'data', 'colmap', 'images.bin')

image_paths = load_image_paths(image_dir, ext='png')
images      = read_images(image_paths, show=False)

# ── 2. Camera intrinsics ─────────────────────
K = get_K_from_colmap(colmap_path)

# ── 3. Feature detection ─────────────────────
sift_data = image_features(images, image_paths, normalize=True)

# ── 4. Feature matching ──────────────────────
all_matches = build_matches(sift_data,
                             ratio_threshold=0.75,
                             min_matches=8,
                             window=1,
                             visualize=False)

# ── 5. Robust F estimation ───────────────────
for match in all_matches:
    i, j = match['image_pair']
    print(f"\nF matrix — Image {i+1} ↔ Image {j+1}")
    F, inliers = ransac_fundamental_matrix(
        match['pts1'], match['pts2'],
        num_iterations=2000,
        threshold=1.5
    )
    match['F']           = F
    match['inliers']     = inliers
    match['pts1_in']     = match['pts1'][inliers]
    match['pts2_in']     = match['pts2'][inliers]
    match['kp_idx1_in']  = match['kp_idx1_raw'][inliers]
    match['kp_idx2_in']  = match['kp_idx2_raw'][inliers]

# ── 6. Essential matrix + pose candidates ────
for match in all_matches:
    i, j = match['image_pair']
    E, poses = essential_matrix(match['F'], K)
    match['E']           = E
    match['K']           = K
    match['poses']       = poses
    match['inlier_pts1'] = match['pts1_in']
    match['inlier_pts2'] = match['pts2_in']
    match['kp_idx1']     = match['kp_idx1_in']
    match['kp_idx2']     = match['kp_idx2_in']

# ── 7. Pose selection + triangulation ────────
camera_poses = process_all_pairs(all_matches, reproj_threshold=2.0)

# ── 8. World-frame trajectory ────────────────
_result        = compute_camera_poses(camera_poses, reverse=True)
positions      = _result[0]
rotations      = _result[1]
transforms_rev = _result[2]
cameras_rev    = _result[3]

plot_camera_trajectory(positions, rotations, cameras_rev,
                       title='Camera Trajectory')

# ── 9. Trajectory comparison vs COLMAP ───────
colmap_pos, estimated_pos = compare_trajectories(
    camera_poses, colmap_path, images_bin
)
compute_baselines(estimated_pos, colmap_pos)

# ── 10. Incremental SfM (mirrors senior's approach) ──
pc = incremental_sfm(
    images,
    K,
    n_features_first=1000,
    n_features_rest=5000,
    ratio=0.75,
    filter_percentile=90
)

points_3d_sfm  = pc.get_points_array()
cam_transforms = pc.camera_poses   # list of 4x4 world-to-cam

# ── 11. Trajectory plot (2-D) ─────────────────
plot_camera_trajectory_labeled(
    [{'R': T[:3, :3], 't': T[:3, 3].reshape(3, 1)} for T in cam_transforms],
    start_cam_num=1
)

# ── 12. 3-D reconstruction ────────────────────
visualize_reconstruction_plotly(
    [{'R': T[:3, :3], 't': T[:3, 3].reshape(3, 1)} for T in cam_transforms],
    points_3d_sfm, start_cam_num=1,
    title='3D Reconstruction — Before Bundle Adjustment',
    metrics=f'{len(points_3d_sfm)} points  |  {len(cam_transforms)} cameras',
    save_path='results/reconstruction_before_ba.html'
)

# ── 13. Bundle adjustment ─────────────────────
cam_tf_dict  = {i: {'R': T[:3, :3], 't': T[:3, 3].reshape(3, 1)}
                for i, T in enumerate(cam_transforms)}
observations = pc.get_observations()

print(f"\nBA input: {len(observations)} observations, "
      f"{len(points_3d_sfm)} points, {len(cam_transforms)} cameras")

# Filter outlier observations before BA
clean_obs = filter_observations(
    observations, points_3d_sfm, cam_tf_dict, K, max_reproj_error=5.0
)
print(f"After outlier filter (<5px): {len(clean_obs)}/{len(observations)}")

# Only keep 3D points seen in at least 2 clean observations
from collections import Counter
pt_obs_count = Counter(pt_idx for _, pt_idx, _ in clean_obs)
valid_pts    = {pt_idx for pt_idx, cnt in pt_obs_count.items() if cnt >= 2}
clean_obs    = [(c, p, m) for c, p, m in clean_obs if p in valid_pts]
print(f"After min-2-obs filter: {len(clean_obs)} obs, {len(valid_pts)} points")

init_errors = compute_reprojection_errors(
    clean_obs, points_3d_sfm, cam_tf_dict, K
)
print_reprojection_stats(init_errors, label='before BA')

cameras_with_obs = {c for c, _, _ in clean_obs}
optimized_points, optimized_poses, ba_result = bundle_adjustment_gtsam(
    clean_obs,
    points_3d_sfm,
    cam_tf_dict,
    K,
    cameras_with_obs,
    num_iterations=200,
    pixel_sigma=3.0,
    verbose=True
)

final_errors = compute_reprojection_errors(
    clean_obs, optimized_points, optimized_poses, K
)
print_reprojection_stats(final_errors, label='after BA')

# Point cloud bounds comparison
for label, pts in [('Before BA', points_3d_sfm), ('After BA', optimized_points)]:
    print(f"\n{label}:")
    for axis, name in enumerate(['X', 'Y', 'Z']):
        print(f"  {name}: [{pts[:, axis].min():.4f}, {pts[:, axis].max():.4f}]")

valid_shifts = np.linalg.norm(
    optimized_points[list(valid_pts)] - points_3d_sfm[list(valid_pts)], axis=1
)
print(f"\nPoint shifts (optimized pts) — "
      f"mean:{valid_shifts.mean():.4f}  "
      f"median:{np.median(valid_shifts):.4f}  "
      f"max:{valid_shifts.max():.4f}")

# ── 14. Visualize before / after BA ──────────
print("\n[1/2] Before Bundle Adjustment:")
visualize_reconstruction_plotly(
    [{'R': T[:3, :3], 't': T[:3, 3].reshape(3, 1)} for T in cam_transforms],
    points_3d_sfm, start_cam_num=1,
    title='3D Reconstruction — Before Bundle Adjustment',
    metrics=(f'{len(points_3d_sfm)} points  |  {len(cam_transforms)} cameras  |  '
             f'Mean reproj: {init_errors.mean():.2f}px  |  '
             f'Max reproj: {init_errors.max():.2f}px'),
    save_path='results/reconstruction_before_ba.html'
)

print("\n[2/2] After Bundle Adjustment:")
optimized_poses_list = [optimized_poses[k]
                        for k in sorted(optimized_poses.keys())]
visualize_reconstruction_plotly(
    optimized_poses_list, optimized_points, start_cam_num=1,
    title='3D Reconstruction — After Bundle Adjustment',
    metrics=(f'{len(optimized_points)} points  |  {len(optimized_poses_list)} cameras  |  '
             f'Mean reproj: {final_errors.mean():.2f}px  |  '
             f'Max reproj: {final_errors.max():.2f}px'),
    save_path='results/reconstruction_after_ba.html'
)