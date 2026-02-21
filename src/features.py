import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def image_features(images, image_paths, normalize=True):
    """
    Detect SIFT and ORB keypoints on every image.
    Only SIFT keypoints + descriptors are stored and returned.
    """
    sift = cv2.SIFT_create()
    orb  = cv2.ORB_create(nfeatures=5000)
    sift_data = []

    for idx, (image, image_path) in enumerate(zip(images, image_paths)):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if normalize:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        kp_sift, des_sift = sift.detectAndCompute(gray, None)
        kp_orb,  des_orb  = orb.detectAndCompute(gray, None)

        # Store pixel coords as array for fast index lookup
        kp_pts = np.array([kp.pt for kp in kp_sift], dtype=np.float32)

        sift_data.append({
            'image_idx':   idx,
            'image':       image,
            'image_path':  image_path,
            'image_shape': image.shape,
            'gray':        gray,
            'keypoints':   kp_sift,
            'kp_pts':      kp_pts,   # (N, 2) pixel coords indexed by kp index
            'descriptors': des_sift
        })

        image_name = os.path.basename(image_path)
        print(f"\n{image_name}")
        print(f"  SIFT: {len(kp_sift)} kp  |  desc: {des_sift.shape if des_sift is not None else 'None'}")
        print(f"  ORB:  {len(kp_orb)}  kp  |  desc: {des_orb.shape  if des_orb  is not None else 'None'}")

        # Visualization
        img_sift = cv2.drawKeypoints(image, kp_sift, None,
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                     color=(0, 255, 0))
        img_orb  = cv2.drawKeypoints(image, kp_orb, None,
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                     color=(255, 0, 0))
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'SIFT: {len(kp_sift)} keypoints', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        axes[1].imshow(cv2.cvtColor(img_orb, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'ORB: {len(kp_orb)} keypoints', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.suptitle(f'Image {idx+1}: {image_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    return sift_data


def sift_features(sift_data1, sift_data2, ratio_threshold=0.75):
    """
    Match SIFT descriptors between two images using Lowe's ratio test.
    Returns matched point arrays, keypoint index arrays, and DMatch list.
    """
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    kp1, des1 = sift_data1['keypoints'], sift_data1['descriptors']
    kp2, des2 = sift_data2['keypoints'], sift_data2['descriptors']

    if des1 is None or des2 is None:
        print("No descriptors found — skipping pair.")
        return None, None, None, None, None

    matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

    if len(good_matches) == 0:
        return None, None, None, None, []

    pts1      = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    pts2      = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
    kp_idx1   = np.array([m.queryIdx for m in good_matches], dtype=np.int32)
    kp_idx2   = np.array([m.trainIdx for m in good_matches], dtype=np.int32)

    return pts1, pts2, kp_idx1, kp_idx2, good_matches


def visualize_sift_matches(sift_data1, sift_data2, good_matches):
    img1, kp1 = sift_data1['image'], sift_data1['keypoints']
    img2, kp2 = sift_data2['image'], sift_data2['keypoints']

    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0)
    )
    name1 = os.path.basename(sift_data1['image_path'])
    name2 = os.path.basename(sift_data2['image_path'])

    plt.figure(figsize=(20, 10))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(f'SIFT Matches: {name1} ↔ {name2}  ({len(good_matches)} matches)',
              fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()