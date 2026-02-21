import cv2
import os
import glob
import struct
import matplotlib.pyplot as plt
import numpy as np


def read_images(images_path, titles=None, show=True):
    images = []
    for idx, image_path in enumerate(images_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return None
        if show:
            title = titles[idx] if titles and idx < len(titles) else f'Image {idx+1}'
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(title)
            plt.axis('off')
            plt.show()
        images.append(image)
    print(f"Loaded {len(images)} images")
    return images


def load_image_paths(image_dir='.', ext='png'):
    paths = sorted(glob.glob(os.path.join(image_dir, f"*.{ext}")))
    print(f"Found {len(paths)} images in '{image_dir}'")
    return paths


def get_K_from_colmap(cameras_bin_path):
    CAMERA_MODELS = {
        0: ('SIMPLE_PINHOLE', 3),
        1: ('PINHOLE', 4),
        2: ('SIMPLE_RADIAL', 4),
        3: ('RADIAL', 5),
        4: ('OPENCV', 8),
    }
    with open(cameras_bin_path, 'rb') as f:
        num_cameras = struct.unpack('Q', f.read(8))[0]
        camera_id   = struct.unpack('I', f.read(4))[0]
        model_id    = struct.unpack('i', f.read(4))[0]
        width       = struct.unpack('Q', f.read(8))[0]
        height      = struct.unpack('Q', f.read(8))[0]

        model_name, num_params = CAMERA_MODELS[model_id]
        params = struct.unpack('d' * num_params, f.read(8 * num_params))

        if model_name == 'PINHOLE':
            fx, fy, cx, cy = params[:4]
        elif model_name == 'SIMPLE_PINHOLE':
            fx = fy = params[0]
            cx, cy  = params[1], params[2]
        elif model_name in ['SIMPLE_RADIAL', 'RADIAL']:
            fx = fy = params[0]
            cx, cy  = params[1], params[2]
        elif model_name == 'OPENCV':
            fx, fy, cx, cy = params[:4]
        else:
            raise ValueError(f"Unsupported camera model: {model_name}")

    K = np.array([
        [fx,  0, cx],
        [0,  fy, cy],
        [0,   0,  1]
    ], dtype=np.float64)

    print(f"Loaded K from COLMAP  |  model: {model_name}  |  size: {width}x{height}")
    print(f"  fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}")
    return K