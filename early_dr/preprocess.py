"""Preprocessing used for every image: crop the black border, pad to a square, resize.

Run once; the processed JPEGs are cached so that training does not repeat this work.
"""
import os
from multiprocessing import Pool

import cv2
import numpy as np


def processed_dir(cfg):
    return os.path.join(cfg.paths.work, f"processed_{cfg.data.prep_size}")


def processed_path(cfg, image_id):
    return os.path.join(processed_dir(cfg), image_id + ".jpg")


def find_crop_box(img, threshold=10):
    """Bounding box (y0, y1, x0, x1) of the retina, i.e. of the non-black pixels."""
    mask = img.mean(axis=2) > threshold
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:          # an all-black image: keep it whole
        return 0, img.shape[0], 0, img.shape[1]
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def crop_pad_resize(img, size, box=None, interpolation=cv2.INTER_AREA):
    """Crop to `box`, pad with black to a square, resize to size x size."""
    y0, y1, x0, x1 = box if box is not None else find_crop_box(img)
    img = img[y0:y1, x0:x1]
    h, w = img.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    canvas = np.zeros((side, side) + img.shape[2:], dtype=img.dtype)
    canvas[top:top + h, left:left + w] = img
    return cv2.resize(canvas, (size, size), interpolation=interpolation)


def _process_one(args):
    src, dst, size = args
    if os.path.exists(dst):
        return True
    img = cv2.imread(src)
    if img is None:                                # corrupt file
        return False
    img = crop_pad_resize(img, size)
    return cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 95])


def preprocess_all(cfg, manifest, workers=4):
    """Process every image in the manifest. Returns the manifest without unreadable images."""
    os.makedirs(processed_dir(cfg), exist_ok=True)
    jobs = [(row.path, processed_path(cfg, row.image_id), cfg.data.prep_size)
            for row in manifest.itertuples()]
    with Pool(workers) as pool:
        ok = pool.map(_process_one, jobs, chunksize=64)
    ok = np.array(ok)
    if (~ok).any():
        print(f"{(~ok).sum()} unreadable images were dropped")
    return manifest[ok].reset_index(drop=True)
