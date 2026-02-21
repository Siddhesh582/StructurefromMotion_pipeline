from src.features import sift_features, visualize_sift_matches


def build_matches(sift_data,
                  ratio_threshold=0.75,
                  min_matches=8,
                  window=1,
                  visualize=True):
    all_matches = []
    n = len(sift_data)

    for i in range(n):
        for j in range(i + 1, min(i + 1 + window, n)):
            print(f"\nMatching Image {i+1} ↔ Image {j+1}")
            pts1, pts2, kp_idx1, kp_idx2, good_matches = sift_features(
                sift_data[i], sift_data[j],
                ratio_threshold=ratio_threshold
            )

            if pts1 is None or len(good_matches) < min_matches:
                print(f"  Skipped — {len(good_matches) if good_matches else 0} "
                      f"matches (need ≥ {min_matches})")
                continue

            all_matches.append({
                'image_pair':  (i, j),
                'pts1':        pts1,
                'pts2':        pts2,
                'kp_idx1_raw': kp_idx1,
                'kp_idx2_raw': kp_idx2,
                'matches':     good_matches,
                'sift_data1':  sift_data[i],
                'sift_data2':  sift_data[j],
            })

            if visualize:
                visualize_sift_matches(sift_data[i], sift_data[j], good_matches)

    print("\n── MATCHING SUMMARY ──")
    for m in all_matches:
        i, j = m['image_pair']
        print(f"  Image {i+1} ↔ Image {j+1} : {len(m['pts1'])} matches")

    return all_matches