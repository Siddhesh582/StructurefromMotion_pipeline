import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go


##

def extract_camera_data(camera_transforms, start_cam_num=1):
    positions, rotations_world = [], []
    for tf in camera_transforms:
        R_w2c = tf['R']
        t_w2c = tf['t'].reshape(3, 1)
        R_c2w = R_w2c.T
        C     = (-R_c2w @ t_w2c).flatten()
        positions.append(C)
        rotations_world.append(R_c2w)
    return np.array(positions), rotations_world


#  2-D trajectory (matplotlib)

def plot_camera_trajectory_labeled(camera_transforms, start_cam_num=3,
                                    save_path='results/pnp_trajectory.png'):
    positions, _ = extract_camera_data(camera_transforms, start_cam_num)
    end_cam_num  = start_cam_num + len(positions) - 1

    fig, ax = plt.subplots(figsize=(14, 12))
    ax.plot(positions[:, 0], positions[:, 1],
            'b-', linewidth=2, alpha=0.6, label='Camera Path')
    ax.scatter(positions[:, 0], positions[:, 1],
               c='blue', s=150, zorder=5, edgecolors='black', linewidths=2)

    ax.scatter(*positions[0,  :2], c='green', s=400, marker='o',
               edgecolors='black', linewidths=2,
               label=f'Start (Cam {start_cam_num})', zorder=10)
    ax.scatter(*positions[-1, :2], c='red',   s=400, marker='o',
               edgecolors='black', linewidths=2,
               label=f'End (Cam {end_cam_num})',   zorder=10)

    for k, pos in enumerate(positions):
        ax.annotate(str(start_cam_num + k), (pos[0], pos[1]),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.4',
                              facecolor='black', alpha=0.8))

    baselines = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    length    = baselines.sum()
    metrics   = (f"Cameras: {len(positions)}  |  "
                 f"Trajectory length: {length:.4f} units  |  "
                 f"Avg baseline: {length/max(len(positions)-1,1):.4f} units")
    ax.set_xlabel('X (world)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y (world)', fontsize=14, fontweight='bold')
    ax.set_title(f'PnP Camera Trajectory  (Cams {start_cam_num}–{end_cam_num})\n{metrics}',
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axis('equal')
    ax.legend(fontsize=11)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved → {save_path}")
    plt.close()


#  3-D reconstruction (Plotly)

def visualize_reconstruction_plotly(camera_transforms, points_3d,
                                     start_cam_num=2,
                                     title=None,
                                     metrics=None,
                                     save_path=None):
    """
    Args:
        camera_transforms : list of {'R', 't'} dicts or dict keyed by cam_idx
        points_3d         : (M, 3) world-frame point cloud
        start_cam_num     : label for first camera
        title             : plot title (shown in browser tab and figure)
        metrics           : optional string of stats shown as subtitle
        save_path         : if set, saves HTML to this path
    """
    # Handle dict input
    if isinstance(camera_transforms, dict):
        tf_list = [camera_transforms[k]
                   for k in sorted(camera_transforms.keys())]
    else:
        tf_list = camera_transforms

    positions, _ = extract_camera_data(tf_list, start_cam_num)

    # Skip cam 1 — it has unit-norm translation from E recovery
    # which places it far from the rest of the trajectory
    positions_plot = np.vstack([positions[0], positions[2:]])

    end_cam_num  = start_cam_num + len(positions) - 1

    fig_title = title or f'3D Reconstruction  (Cams {start_cam_num}–{end_cam_num})'
    if metrics:
        fig_title += f'<br><sup>{metrics}</sup>'

    traces = []

    # Point cloud
    if len(points_3d) > 0:
        traces.append(go.Scatter3d(
            x=points_3d[:, 0], y=points_3d[:, 1], z=points_3d[:, 2],
            mode='markers',
            marker=dict(size=0.7, color='brown', opacity=1.0),
            name='3D Points',
            hovertemplate='X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
        ))

    # Camera positions (unique only)
    traces.append(go.Scatter3d(
        x=positions_plot[:, 0], y=positions_plot[:, 1], z=positions_plot[:, 2],
        mode='markers',
        marker=dict(size=3, color='blue', opacity=0.8),
        name='Cameras',
        hovertemplate='Cam %{text}<extra></extra>',
        text=[str(start_cam_num + k) for k in range(len(positions_plot))]
    ))

    # Start / end
    traces.append(go.Scatter3d(
        x=[positions_plot[0,  0]], y=[positions_plot[0,  1]], z=[positions_plot[0,  2]],
        mode='markers',
        marker=dict(size=6, color='green', symbol='diamond',
                    line=dict(color='black', width=2)),
        name=f'Start (Cam {start_cam_num})'
    ))
    traces.append(go.Scatter3d(
        x=[positions_plot[-1, 0]], y=[positions_plot[-1, 1]], z=[positions_plot[-1, 2]],
        mode='markers',
        marker=dict(size=6, color='orange', symbol='diamond',
                    line=dict(color='black', width=2)),
        name=f'End (Cam {end_cam_num})'
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=fig_title, font=dict(size=16)),
        scene=dict(
            xaxis_title='X (world)',
            yaxis_title='Y (world)',
            zaxis_title='Z (world)',
            aspectmode='data'
        ),
        width=1400, height=900,
        legend=dict(itemsizing='constant')
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        print(f"  Saved → {save_path}")

    fig.show()
    _print_trajectory_stats(positions, start_cam_num, end_cam_num,
                             n_points=len(points_3d))


##

def _print_trajectory_stats(positions, start_cam, end_cam, n_points=None):
    baselines = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    length    = baselines.sum()
    n_cams    = len(positions)
    print(f"\nTrajectory Statistics:")
    print(f"  Cameras       : {n_cams}  (Cam {start_cam} – Cam {end_cam})")
    print(f"  Total length  : {length:.4f} units")
    print(f"  Avg baseline  : {length / max(n_cams - 1, 1):.4f} units")
    if n_points is not None:
        print(f"  3D points     : {n_points}")