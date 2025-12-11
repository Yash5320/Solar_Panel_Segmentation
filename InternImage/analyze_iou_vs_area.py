import sys
import os
# Add the 'segmentation' folder to Python's path to find custom modules.
sys.path.insert(0, './segmentation')

import mmseg_custom # noqa: F401,F403
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from mmseg.apis import init_segmentor, inference_segmentor
from mmcv import Config

# --- Configuration ---
config_file = 'segmentation/configs/ade20k/exp_v5_lr_aug.py'
checkpoint_file = 'segmentation/work_dirs/exp_v5_aug/best_mIoU_iter_160000.pth'
output_filename = 'iou_vs_area_scatter.png'

# --- IMPORTANT: Set the correct paths to your validation data ---
val_img_dir = '/mnt/HDD10/ydeole/val/img/'
val_mask_dir = '/mnt/HDD10/ydeole/val/mask/'
# --- End of Configuration ---

# Build the model from a config file and a checkpoint file
# Use 'cuda:0' to specify the first available GPU
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# Get a list of all validation images
image_filenames = [f for f in os.listdir(val_img_dir) if f.endswith('.png')]
print(f"Found {len(image_filenames)} images in {val_img_dir}")

area_iou_pairs = []
print("Evaluating validation set to get per-image IoU and area...")

# Loop through each image file
for filename in tqdm(image_filenames):
    img_path = os.path.join(val_img_dir, filename)
    mask_path = os.path.join(val_mask_dir, filename)

    # 1. Load the ground truth mask
    # We use cv2.IMREAD_GRAYSCALE to load it as a single channel image
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        continue # Skip if mask is not found

    # 2. Run inference on the image to get the prediction
    # inference_segmentor handles all the necessary preprocessing
    result = inference_segmentor(model, img_path)
    pred_mask = result[0] # The result is a numpy array

    # 3. Calculate Ground Truth Area for the solar panel class (pixel value 1)
    gt_area = np.sum(gt_mask == 1)
    if gt_area == 0:
        continue # Skip images with no solar panels in the ground truth

    # 4. Calculate IoU for the solar panel class
    intersection = np.sum((pred_mask == 1) & (gt_mask == 1))
    union = np.sum((pred_mask == 1) | (gt_mask == 1))
    iou = intersection / union if union > 0 else 0.0
    
    area_iou_pairs.append((gt_area, iou))

if not area_iou_pairs:
    print("Error: No valid data was processed. Check your val_img_dir and val_mask_dir paths.")
    exit()

# Create the scatter plot
areas = [p[0] for p in area_iou_pairs]
ious = [p[1] for p in area_iou_pairs]

plt.figure(figsize=(12, 6))
plt.scatter(areas, ious, alpha=0.5)
plt.xscale('log')
plt.xlabel('Ground Truth Area (pixels, log scale)')
plt.ylabel('IoU')
plt.title('IoU vs. Ground Truth Solar Panel Area')
plt.grid(True, which="both", ls="--")
plt.savefig(output_filename)
print(f"\nSuccessfully generated scatter plot: '{output_filename}'")
