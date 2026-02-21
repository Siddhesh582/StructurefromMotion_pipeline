import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_camera_poses(camera_poses, reverse=False):
    """
    Chain relative poses into a world-frame trajectory.

    Args:
        camera_poses : list of pose dicts (must have 'selected_R', 'selected_t')
        reverse      : if True, chain in reverse order then flip to normal ordering

    Returns:
        positions              : (N, 3) world-frame camera centres
        camera_rotations_world : list of N rotation matrices (cam-to-world)
        camera_transforms      : list of N dicts with world-to-cam {R, t}
        cameras_used           : list of image indices
    """
    positions_world = [np.zeros(3)]
    rotations_world = [np.eye(3)]
    transforms      = [{'R': np.eye(3), 't': np.zeros((3, 1))}]
    cameras_used    = [0]

    R_w2c = np.eye(3)
    t_w2c = np.zeros((3, 1))

    direction = "REVERSE" if reverse else "FORWARD"
    print(f"\nBuilding trajectory — {direction}")
    print(f"  {'Pair':<10} {'t_rel':<38} {'Cam Pos (world)'}")
    print("─" * 90)

    valid_pairs = [
        (idx, p) for idx, p in enumerate(camera_poses)
        if 'selected_R' in p and p['image_pair'][1] == p['image_pair'][0] + 1
    ]

    iterator = reversed(valid_pairs) if reverse else iter(valid_pairs)

    for idx, pose_data in iterator:
        i, j  = pose_data['image_pair']
        R_fwd = pose_data['selected_R']
        t_fwd = pose_data['selected_t'].reshape(3, 1)

        if reverse:
            R_rel = R_fwd.T
            t_rel = -R_rel @ t_fwd
            label = f"{j+1}←{i+1}"
        else:
            R_rel = R_fwd
            t_rel = t_fwd
            label = f"{i+1}→{j+1}"

        t_w2c = R_rel @ t_w2c + t_rel
        R_w2c = R_rel @ R_w2c

        transforms.append({'R': R_w2c.copy(), 't': t_w2c.copy()})

        cam_pos = (-R_w2c.T @ t_w2c).flatten()
        cam_rot = R_w2c.T

        positions_world.append(cam_pos)
        rotations_world.append(cam_rot)
        cameras_used.append(j if not reverse else i)

        print(f"  {label:<10} "
              f"[{t_rel[0,0]:6.3f}, {t_rel[1,0]:6.3f}, {t_rel[2,0]:6.3f}]   "
              f"[{cam_pos[0]:7.3f}, {cam_pos[1]:7.3f}, {cam_pos[2]:7.3f}]")

    if reverse:
        positions_world = list(reversed(positions_world))
        rotations_world = list(reversed(rotations_world))
        transforms      = list(reversed(transforms))
        cameras_used    = list(reversed(cameras_used))
        print("\n  Reversed to normal order (Camera 1 → N)")

    print(f"\n  Cameras in trajectory : {len(cameras_used)}")
    print(f"  Indices               : {cameras_used}")

    return np.array(positions_world), rotations_world, transforms, cameras_used


def plot_camera_trajectory(positions, rotations_world, cameras_used=None,
                            title='Camera Trajectory',
                            save_path='results/trajectory.png'):
    """2-D top-down (X-Y) view of the camera trajectory with viewing directions."""
    fig, ax = plt.subplots(figsize=(14, 12))

    ax.plot(positions[:, 0], positions[:, 1],
            'b-', linewidth=2, alpha=0.6, label='Camera path')
    ax.scatter(positions[:, 0], positions[:, 1],
               c='blue', s=150, zorder=5, edgecolors='black', linewidths=2)

    arrow_len = 0.5
    for pos, R in zip(positions, rotations_world):
        d = R @ np.array([0, 0, 1])
        ax.arrow(pos[0], pos[1], d[0]*arrow_len, d[1]*arrow_len,
                 head_width=0.15, head_length=0.1,
                 fc='red', ec='red', alpha=0.7, linewidth=1.5)

    ax.scatter(*positions[0,  :2], c='green', s=400, marker='o',
               edgecolors='black', linewidths=2, label='Start', zorder=10)
    ax.scatter(*positions[-1, :2], c='red',   s=400, marker='o',
               edgecolors='black', linewidths=2, label='End',   zorder=10)

    for k, pos in enumerate(positions):
        lbl = cameras_used[k] + 1 if cameras_used else k + 1
        ax.annotate(str(lbl), (pos[0], pos[1]),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.4',
                              facecolor='black', alpha=0.8))

    ax.set_xlabel('X (world)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y (world)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axis('equal')
    ax.legend(fontsize=11)
    plt.tight_layout()

    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved trajectory plot → {save_path}")
    plt.close()