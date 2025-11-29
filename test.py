import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from helpers.preprocess import preprocess_images

print("TF version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

# ---- test: show before / after for a single image ----
image_rel = "data/images/00000001_000.png"
image_path = os.path.join(os.path.dirname(__file__), image_rel)

if not os.path.exists(image_path):
    print("Image not found:", image_path)
else:
    # load original
    orig = Image.open(image_path).convert("RGB")
    orig_np = np.array(orig)

    # plot original
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(orig_np)
    axs[0].set_title("Original")
    axs[0].axis("off")

    # run preprocessing which will overwrite the file and return normalized array
    try:
        processed_stack = preprocess_images([image_path])  # returns (1,200,200,3) float32 in [0,1]
        if processed_stack.shape[0] == 1:
            after_np = (processed_stack[0] * 255.0).astype("uint8")
        else:
            after_np = np.array(Image.open(image_path).convert("RGB"))
    except Exception as e:
        print("Preprocess failed:", e)
        after_np = np.array(Image.open(image_path).convert("RGB"))

    axs[1].imshow(after_np)
    axs[1].set_title("After preprocessing (200x200)")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()
