import struct
import numpy as np
import matplotlib.pyplot as plt

from src.cam_trajectory import compute_camera_poses  # noqa: E402


# ─────────────────────────────────────────────
#  Quaternion → Rotation Matrix
# ─────────────────────────────────────────────

def quat_to_rotation_matrix(qw, qx, qy, qz):
    """Convert a unit quaternion (COLMAP convention) to a 3×3 rotation matrix."""
    R = np.array([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qw*qz),   2*(qx*qz + qw*qy)],
        [    2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2),  2*(qy*qz - qw*qx)],
        [    2*(qx*qz - qw*qy),   2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)],
    ])
    return R


# ─────────────────────────────────────────────
#  Read COLMAP images.bin
# ─────────────────────────────────────────────

def read_colmap_images(images_bin_path):
    """
    Parse COLMAP images.bin.
    Returns:
        colmap_positions : (N, 3) camera centres in world frame
        colmap_rotations : list of N cam-to-world rotation matrices
    """
    positions = []
    rotations = []

    with open(images_bin_path, 'rb') as f:
        num_images = struct.unpack('Q', f.read(8))[0]

        for _ in range(num_images):
            struct.unpack('I', f.read(4))           # image_id (unused)
            qw, qx, qy, qz = struct.unpack('4d', f.read(32))
            tx, ty, tz      = struct.unpack('3d', f.read(24))
            struct.unpack('I', f.read(4))           # camera_id (unused)

            # Read null-terminated image name
            name = b''
            while True:
                ch = f.read(1)
                if ch == b'\x00':
                    break
                name += ch

            num_pts2d = struct.unpack('Q', f.read(8))[0]
            f.read(24 * num_pts2d)                  # skip 2D point records

            R      = quat_to_rotation_matrix(qw, qx, qy, qz)
            t_vec  = np.array([tx, ty, tz]).reshape(3, 1)
            cam_pos = (-R.T @ t_vec).flatten()

            positions.append(cam_pos)
            rotations.append(R.T)                   # cam-to-world

    positions = np.array(positions)
    # Shift so first camera is at origin (same as estimated trajectory)
    offset    = positions[0].copy()
    positions = positions - offset
    print(f"COLMAP: {len(positions)} cameras loaded  "
          f"(origin offset: [{offset[0]:.3f}, {offset[1]:.3f}, {offset[2]:.3f}])")
    return positions, rotations


# ─────────────────────────────────────────────
#  Trajectory Comparison Plot
# ─────────────────────────────────────────────

def compare_trajectories(camera_poses, cameras_bin_path, images_bin_path):
    """
    Overlay COLMAP ground-truth and estimated trajectories on a 2-D top-down plot.
    Returns (colmap_positions, estimated_positions).
    """
    colmap_positions, colmap_rotations = read_colmap_images(images_bin_path)

    est_positions, est_rotations, _, est_cameras_used = \
        compute_camera_poses(camera_poses, reverse=True)

    fig, ax = plt.subplots(figsize=(14, 12))

    # ── COLMAP ──
    ax.plot(colmap_positions[:, 0], colmap_positions[:, 1],
            'g-', linewidth=2.5, alpha=0.7, label='COLMAP (Ground Truth)')
    ax.scatter(colmap_positions[:, 0], colmap_positions[:, 1],
               c='green', s=120, zorder=5, edgecolors='black', linewidths=2, alpha=0.7)

    # ── Estimated ──
    ax.plot(est_positions[:, 0], est_positions[:, 1],
            'b--', linewidth=2.5, alpha=0.7, label='Estimated')
    ax.scatter(est_positions[:, 0], est_positions[:, 1],
               c='blue', s=120, zorder=6, edgecolors='black', linewidths=2, alpha=0.7)

    ax.scatter(0, 0, c='red', s=400, marker='o',
               edgecolors='black', linewidths=3, label='Start (Origin)', zorder=10)

    ax.set_xlabel('X (world)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y (world)', fontsize=14, fontweight='bold')
    ax.set_title('Trajectory Comparison: COLMAP vs Estimated',
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

    # ── COLMAP position table ──
    print("\nCOLMAP Camera Positions:")
    print(f"{'Cam':<6} {'X':>8} {'Y':>8} {'Z':>8}")
    print("─" * 36)
    for k, pos in enumerate(colmap_positions):
        print(f"{k+1:<6} {pos[0]:8.3f} {pos[1]:8.3f} {pos[2]:8.3f}")

    return colmap_positions, est_positions


# ─────────────────────────────────────────────
#  Baseline Analysis
# ─────────────────────────────────────────────

def _baseline_stats(positions, label):
    """Compute and print sequential baselines for a trajectory."""
    baselines = np.linalg.norm(np.diff(positions, axis=0), axis=1)

    print(f"\n{'='*60}")
    print(f"BASELINES — {label}")
    print(f"{'='*60}")
    print(f"{'Pair':<10} {'Baseline':>12}  units")
    print("─" * 60)
    for k, b in enumerate(baselines):
        print(f"{k+1}→{k+2:<7}  {b:12.4f}")
    print("─" * 60)
    print(f"Total length : {baselines.sum():.4f}")
    print(f"Mean baseline: {baselines.mean():.4f}")
    print(f"Min  baseline: {baselines.min():.4f}  "
          f"(pair {baselines.argmin()+1}→{baselines.argmin()+2})")
    print(f"Max  baseline: {baselines.max():.4f}  "
          f"(pair {baselines.argmax()+1}→{baselines.argmax()+2})")
    print(f"{'='*60}")
    return baselines


def compute_baselines(estimated_positions, colmap_positions=None):
    """
    Print baseline stats for the estimated trajectory and optionally COLMAP.
    Returns (est_baselines, colmap_baselines) — colmap_baselines is None if
    colmap_positions is not provided.
    """
    est_bl    = _baseline_stats(estimated_positions, "Estimated")
    colmap_bl = None
    if colmap_positions is not None:
        colmap_bl = _baseline_stats(colmap_positions, "COLMAP (Ground Truth)")
    return est_bl, colmap_bl