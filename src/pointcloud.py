import numpy as np


class PointCloud:
    """
    Incremental 3D point cloud with feature track management.
    tracks: { tuple(X_3d) : [(cam_idx, tuple(pt_2d)), ...] }
    """

    def __init__(self):
        self.tracks       = {}
        self.camera_poses = []   # list of 4x4 world-to-cam transforms

    def add_points(self, pts_3d, img1_pts, img2_pts, cam_idx1, cam_idx2):
        """
        Add triangulated 3D points and their 2D observations.
        Merges with existing tracks if a matching observation is found.
        """
        for pt3d, p1, p2 in zip(pts_3d, img1_pts, img2_pts):
            obs1 = (cam_idx1, tuple(p1))
            obs2 = (cam_idx2, tuple(p2))
            key  = tuple(pt3d)

            merged = False
            for track_key, obs_list in self.tracks.items():
                if obs1 in obs_list or obs2 in obs_list:
                    if obs1 not in obs_list:
                        obs_list.append(obs1)
                    if obs2 not in obs_list:
                        obs_list.append(obs2)
                    merged = True
                    break

            if not merged:
                self.tracks[key] = [obs1, obs2]

        print(f"  Tracks: {len(self.tracks)}")

    def common_pts(self, img1_pts, img2_pts):
        """
        Find 3D points already observed in the current source frame.
        Matches img1_pts by pixel coordinate to stored observations,
        returns the corresponding 3D points and img2_pts for PnP.
        """
        matched_3d, matched_2d = [], []

        for i, pt in enumerate(img1_pts):
            pt_key = tuple(pt)
            for track_key, obs_list in self.tracks.items():
                stored_pts = [obs[1] for obs in obs_list]
                if pt_key in stored_pts:
                    matched_3d.append(np.array(track_key))
                    matched_2d.append(img2_pts[i])
                    break

        if len(matched_3d) == 0:
            return np.empty((0, 3)), np.empty((0, 2))

        return np.array(matched_3d), np.array(matched_2d)

    def filter_points(self, percentile=90):
        """
        Remove points beyond the given depth percentile.
        Keeps the bottom `percentile`% by Z value.
        """
        if not self.tracks:
            return

        pts    = np.array(list(self.tracks.keys()))
        depths = pts[:, 2]
        thresh = np.percentile(depths, percentile)

        to_remove = [k for k in self.tracks if k[2] > thresh]
        for k in to_remove:
            del self.tracks[k]

        print(f"  filter_points({percentile}%): "
              f"removed {len(to_remove)}, kept {len(self.tracks)}")

    def get_points_array(self):
        if not self.tracks:
            return np.empty((0, 3))
        return np.array(list(self.tracks.keys()))

    def get_observations(self):
        """Flat list of (cam_idx, pt_idx, 2d_pt) for bundle adjustment."""
        observations = []
        for pt_idx, (_, obs_list) in enumerate(self.tracks.items()):
            for cam_idx, pt_2d in obs_list:
                observations.append((cam_idx, pt_idx, np.array(pt_2d)))
        return observations

    def __len__(self):
        return len(self.tracks)