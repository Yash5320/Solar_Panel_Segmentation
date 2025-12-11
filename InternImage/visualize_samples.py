import sys
# THIS IS THE FIX: Add the 'segmentation' folder to Python's path
# so it can find the custom modules.
sys.path.insert(0, './segmentation')

import os
import cv2
import mmseg_custom # noqa: F401,F403
import numpy as np
import matplotlib.pyplot as plt
from mmseg.apis import init_segmentor, inference_segmentor
from mmcv import Config

# --- Configuration ---
config_file = 'segmentation/configs/ade20k/exp_v5_lr_aug.py'
checkpoint_file = 'segmentation/work_dirs/exp_v5_aug/best_mIoU_iter_160000.pth'

# List of image basenames to visualize (without .png)
image_names = [
    'ZLVSC7C2CEKGIW',  # Small
    'STFNN6D12IUTJU',  # Medium (irregular)
    'IJYJF237BDLZX'   # Large
]

# --- IMPORTANT: Set the correct paths to your validation data ---
val_img_dir = '/mnt/HDD10/ydeole/val/img/'
val_mask_dir = '/mnt/HDD10/ydeole/val/mask/'
output_dir = 'visualizations/'
# --- End of Configuration ---

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

for name in image_names:
    print(f"Processing: {name}")
    img_path = os.path.join(val_img_dir, f"{name}.png")
    mask_path = os.path.join(val_mask_dir, f"{name}.png")

    # Load raw image and ground truth
    raw_image = cv2.imread(img_path)
    raw_image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    ground_truth_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Run inference to get the prediction
    result = inference_segmentor(model, img_path)
    prediction_mask = result[0].astype(np.uint8)

    # Create the 3-panel plot
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle(f'Sample: {name}', fontsize=20)

    axes[0].imshow(raw_image_rgb)
    axes[0].set_title('Raw Image', fontsize=16)
    axes[0].axis('off')

    axes[1].imshow(ground_truth_mask, cmap='gray')
    axes[1].set_title('Ground Truth', fontsize=16)
    axes[1].axis('off')

    axes[2].imshow(prediction_mask, cmap='gray')
    axes[2].set_title('Model Prediction', fontsize=16)
    axes[2].axis('off')

    output_path = os.path.join(output_dir, f"{name}_comparison.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {output_path}")
