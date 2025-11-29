import numpy as np
from PIL import Image
from .segment import optimal_thresholding

def center_crop_or_pad(img_array, target_size=(200, 200)):
    th, tw = target_size
    h, w = img_array.shape[:2]
    channels = img_array.shape[2] if img_array.ndim == 3 else 1

    # If already exact size, return as-is
    if (h, w) == (th, tw):
        return img_array

    # Start with a black canvas
    if channels == 1:
        canvas = np.zeros((th, tw), dtype=img_array.dtype)
    else:
        canvas = np.zeros((th, tw, channels), dtype=img_array.dtype)

    # Compute copy coordinates (centered)
    y_start = max((th - h) // 2, 0)
    x_start = max((tw - w) // 2, 0)
    y_end = y_start + min(h, th)
    x_end = x_start + min(w, tw)

    src_y_start = max((h - th) // 2, 0)
    src_x_start = max((w - tw) // 2, 0)
    src_y_end = src_y_start + (y_end - y_start)
    src_x_end = src_x_start + (x_end - x_start)

    if channels == 1:
        canvas[y_start:y_end, x_start:x_end] = img_array[src_y_start:src_y_end, src_x_start:src_x_end]
    else:
        canvas[y_start:y_end, x_start:x_end, :] = img_array[src_y_start:src_y_end, src_x_start:src_x_end, :]

    return canvas

# Preprocess images
def preprocess_images(image_paths):
    processed = []
    count = 0
    for path in image_paths:
        count += 1
        print(f"Preprocessing image {count}/{len(image_paths)}: {path}")
        img = Image.open(path).convert("RGB")
        img_np = np.array(img)  # HxWx3 uint8

        # segmentation mask (grayscale internally)
        mask, thr, it = optimal_thresholding(img_np)
        # mask is HxW boolean where True currently indicates outer/non-lung region
        # Previously we used masked[~mask] = 0 which removed lungs when mask marked outer.
        # Fix: remove outer region (mask==True)
        masked = img_np.copy()
        masked[mask] = 0

        # center crop or pad to 200x200 (if already 200x200, returned unchanged)
        final = center_crop_or_pad(masked, (200, 200))

        # Save/overwrite original file (preserve format by extension)
        try:
            save_img = Image.fromarray(final)
            save_img.save(path)
        except Exception:
            # fallback: convert to uint8 RGB and overwrite as PNG
            Image.fromarray(final.astype(np.uint8)).save(path)

        # normalize to [0,1] float32 for return
        processed.append(final.astype(np.float32) / 255.0)

    return np.stack(processed, axis=0) if processed else np.empty((0, 200, 200, 3), dtype=np.float32)
