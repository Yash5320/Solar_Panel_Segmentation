import sys
import os

# --- CRITICAL FIX: Setup path and import custom modules FIRST ---
# 1. Add the 'segmentation' folder to Python's path
sys.path.insert(0, './segmentation')

# 2. Import mmseg_custom to register 'InternImage' in the model registry
try:
    import mmseg_custom 
except ImportError as e:
    print(f"Error importing mmseg_custom: {e}")
    print("Please ensure you are running this script from the root 'InternImage' directory.")
    sys.exit(1)
# ----------------------------------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mmseg.apis import init_segmentor, inference_segmentor

# --- Configuration ---
config_file = 'segmentation/configs/ade20k/exp_v5_lr_aug.py'
checkpoint_file = 'segmentation/work_dirs/exp_v5_aug/best_mIoU_iter_160000.pth'
plot_output_filename = 'UCI_Campus_Test_iou_vs_area.png'

# --- Paths ---

#test_img_dir = '/mnt/HDD10/ydeole/UCI_Campus_Test/img/'
#test_mask_dir = '/mnt/HDD10/ydeole/UCI_Campus_Test/mask/'
#save_pred_dir = '/mnt/HDD10/ydeole/UCI_Campus_Test/InternImagePreds/'

test_img_dir = '/mnt/HDD10/ydeole/independent_test/img/'
test_mask_dir = '/mnt/HDD10/ydeole/independent_test/mask/'
save_pred_dir = '/mnt/HDD10/ydeole/independent_test/InternImagePreds/'

# Ensure save directory exists
os.makedirs(save_pred_dir, exist_ok=True)

# --- Target Class Definition ---
TARGET_CLASS_ID = 1 

def calculate_binary_metrics(pred_mask, gt_mask, target_id=1):
    p = (pred_mask == target_id).astype(np.uint8)
    g = (gt_mask == target_id).astype(np.uint8)
    tp = np.sum((p == 1) & (g == 1))
    tn = np.sum((p == 0) & (g == 0))
    fp = np.sum((p == 1) & (g == 0))
    fn = np.sum((p == 0) & (g == 1))
    return tp, tn, fp, fn

def compute_scores(tp, tn, fp, fn):
    union = tp + fp + fn
    iou = tp / union if union > 0 else 0.0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return iou, dice, precision, recall, accuracy

def main():
    print("Initializing model...")
    # Initialize the segmentor (Backbone 'InternImage' is now registered via the import above)
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

    image_filenames = [f for f in os.listdir(test_img_dir) if f.endswith('.png')]
    image_filenames.sort()
    
    print(f"Found {len(image_filenames)} images.")
    print(f"Predictions will be saved to: {save_pred_dir}")

    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
    results_list = []

    for filename in tqdm(image_filenames):
        img_path = os.path.join(test_img_dir, filename)
        mask_filename = filename.replace('.png', '_mask.png')
        mask_path = os.path.join(test_mask_dir, mask_filename)

        if not os.path.exists(mask_path):
            continue

        # 1. Load GT and FIX VALUES (Convert 255 -> 1)
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            continue
        gt_mask[gt_mask > 0] = 1 

        # 2. Inference
        result = inference_segmentor(model, img_path)
        pred_mask = result[0].astype(np.uint8)

        # 3. Save Prediction (Convert 1 -> 255 for visibility)
        save_path = os.path.join(save_pred_dir, mask_filename)
        cv2.imwrite(save_path, pred_mask * 255)

        # 4. Metrics
        tp, tn, fp, fn = calculate_binary_metrics(pred_mask, gt_mask, TARGET_CLASS_ID)
        
        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn

        iou, dice, prec, rec, acc = compute_scores(tp, tn, fp, fn)
        gt_area = tp + fn 

        results_list.append({
            'area': gt_area,
            'iou': iou,
            'dice': dice,
            'precision': prec,
            'recall': rec,
            'accuracy': acc
        })

    if not results_list:
        print("No images processed.")
        return

    # Averages
    m_iou = np.mean([r['iou'] for r in results_list])
    
    # Global
    g_iou, g_dice, g_prec, g_rec, g_acc = compute_scores(total_tp, total_tn, total_fp, total_fn)

    print("\n" + "="*40)
    print("       EVALUATION RESULTS")
    print("="*40)
    print(f"Mean IoU (mIoU): {m_iou:.4f}")
    print(f"Global IoU:      {g_iou:.4f}")
    print("-" * 40)
    print(f"Global Precision: {g_prec:.4f}")
    print(f"Global Recall:    {g_rec:.4f}")
    print(f"Global F1-Score:  {g_dice:.4f}")
    print("="*40)
    print("Confusion Matrix:")
    print(f"TP: {total_tp} | FP: {total_fp}")
    print(f"FN: {total_fn} | TN: {total_tn}")
    print("="*40)

    # Plot
    valid_plot_data = [r for r in results_list if r['area'] > 0]
    areas = [r['area'] for r in valid_plot_data]
    ious = [r['iou'] for r in valid_plot_data]

    if valid_plot_data:
        plt.figure(figsize=(12, 6))
        plt.scatter(areas, ious, alpha=0.6)
        plt.xscale('log')
        plt.xlabel('Ground Truth Area (pixels)')
        plt.ylabel('IoU')
        plt.title('IoU vs Area')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.savefig(plot_output_filename)
        print(f"Plot saved to {plot_output_filename}")
    else:
        print("Not enough data points to generate scatter plot.")

if __name__ == '__main__':
    main()
