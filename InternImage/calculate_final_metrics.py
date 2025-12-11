import sys
import os
# Add the current directory to Python's path to find custom modules.
sys.path.insert(0, os.getcwd()) 

import mmseg_custom # noqa: F401,F403
import torch
import numpy as np
import cv2
from tqdm import tqdm
from mmseg.apis import init_segmentor, inference_segmentor
from mmcv import Config

# --- Configuration ---
config_file = 'configs/ade20k/exp_v5_lr_aug.py'
checkpoint_file = 'work_dirs/exp_v5_aug/best_mIoU_iter_160000.pth'

# --- IMPORTANT: Set the correct paths to your validation data ---
val_img_dir = '/mnt/HDD10/ydeole/val/img/'
val_mask_dir = '/mnt/HDD10/ydeole/val/mask/'
# --- End of Configuration ---

# Build the model
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# Get a list of all validation images
image_filenames = [f for f in os.listdir(val_img_dir) if f.endswith('.png')]
print(f"Found {len(image_filenames)} images in {val_img_dir}")

# Initialize counters for the entire dataset
total_true_positives = 0
total_false_positives = 0
total_false_negatives = 0
total_intersection = 0
total_union = 0

print("Calculating metrics over the entire validation set...")

# Loop through each image file
for filename in tqdm(image_filenames):
    img_path = os.path.join(val_img_dir, filename)
    mask_path = os.path.join(val_mask_dir, filename)

    # Load ground truth mask
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        continue

    # Run inference to get the prediction
    result = inference_segmentor(model, img_path)
    pred_mask = result[0]

    # Calculate metrics for the solar panel class (pixel value 1)
    true_positives = np.sum((pred_mask == 1) & (gt_mask == 1))
    false_positives = np.sum((pred_mask == 1) & (gt_mask == 0))
    false_negatives = np.sum((pred_mask == 0) & (gt_mask == 1))
    
    intersection = true_positives
    union = true_positives + false_positives + false_negatives

    # Accumulate totals
    total_true_positives += true_positives
    total_false_positives += false_positives
    total_false_negatives += false_negatives
    total_intersection += intersection
    total_union += union

# --- Final Metric Calculations ---
# Add a small epsilon to avoid division by zero
epsilon = 1e-6

# Overall IoU for the solar panel class
iou_solar_panel = total_intersection / (total_union + epsilon)

# Overall Precision
precision = total_true_positives / (total_true_positives + total_false_positives + epsilon)

# Overall Recall
recall = total_true_positives / (total_true_positives + total_false_negatives + epsilon)

# Overall F1 Score
f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

# Print the final results in a clean table
print("\n--- Final Performance Metrics ---")
print(f"Solar Panel IoU: {iou_solar_panel * 100:.2f}%")
print(f"Precision:       {precision * 100:.2f}%")
print(f"Recall:          {recall * 100:.2f}%")
print(f"F1 Score:        {f1_score * 100:.2f}%")
print("---------------------------------")
