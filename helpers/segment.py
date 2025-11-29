import numpy as np
from skimage import color, exposure
from skimage.measure import label, regionprops
from skimage.morphology import binary_opening

def optimal_thresholding(image, initial_threshold=0.72, max_iterations=100, tolerance=1e-4):
    if image is None:
        raise ValueError("image is None")

    # Accept RGB or grayscale numpy arrays
    if image.ndim == 3 and image.shape[2] in (3, 4):
        gray = color.rgb2gray(image[..., :3])
    else:
        gray = image.astype(np.float32)

    # Normalize to [0,1]
    gray = exposure.rescale_intensity(gray, out_range=(0.0, 1.0))

    threshold = float(initial_threshold)
    prev_threshold = threshold + 2 * tolerance
    iteration = 0

    while abs(threshold - prev_threshold) > tolerance and iteration < max_iterations:
        prev_threshold = threshold
        lung_voxels = gray[gray <= threshold]
        non_lung_voxels = gray[gray > threshold]

        if lung_voxels.size > 0 and non_lung_voxels.size > 0:
            mu_a = lung_voxels.mean()
            mu_b = non_lung_voxels.mean()
            threshold = (mu_a + mu_b) / 2.0
        else:
            break

        iteration += 1

    mask = (gray >= threshold)  # boolean mask
    return mask, float(threshold), int(iteration)

def remove_outer_region(image):
    # Accept boolean mask or 0/255 uint8 images
    if image.dtype == bool:
        black_mask = ~image
    else:
        black_mask = (image == 0)

    labeled_img = label(black_mask)
    height, width = labeled_img.shape

    border_labels = set()
    border_labels.update(np.unique(labeled_img[0, :]))
    border_labels.update(np.unique(labeled_img[-1, :]))
    border_labels.update(np.unique(labeled_img[:, 0]))
    border_labels.update(np.unique(labeled_img[:, -1]))
    border_labels.discard(0)

    cleaned = np.ones_like(labeled_img, dtype=np.uint8) * 255
    if border_labels:
        cleaned[np.isin(labeled_img, list(border_labels))] = 0

    return cleaned  # uint8 0/255

def reconstruct(slice_prev, slice_curr, slice_next):
    # compute absolute combined difference
    prev = slice_prev.astype(np.int16)
    curr = slice_curr.astype(np.int16)
    nxt = slice_next.astype(np.int16)

    difference = np.abs(prev - curr) + np.abs(nxt - curr)
    difference = np.clip(difference, 0, 255).astype(np.uint8)

    if difference.ndim == 3:
        # work on grayscale magnitude if color
        diff_gray = color.rgb2gray(difference[..., :3])
        binary_mask = diff_gray > 0
    else:
        binary_mask = difference > 0

    # structuring element shaped for 2D masks
    struct_elem = np.ones((2, 2), dtype=bool)
    cleaned_diff = binary_opening(binary_mask, structure=struct_elem)

    labeled_img = label(cleaned_diff)
    valid_regions = np.zeros_like(labeled_img, dtype=bool)
    for region in regionprops(labeled_img):
        if 10 <= region.area <= 300:
            perimeter = region.perimeter if region.perimeter > 0 else 1.0
            circularity = (4.0 * np.pi * region.area) / (perimeter ** 2)
            if circularity > 0.3:
                valid_regions[labeled_img == region.label] = True

    return (valid_regions * 255).astype(np.uint8)

def logical_and(optimal_threshold_mask, largest_region_mask):
    # Normalize inputs to boolean masks
    a = np.asarray(optimal_threshold_mask).astype(bool)
    b = np.asarray(largest_region_mask).astype(bool)
    return np.logical_and(np.logical_not(a), b)
