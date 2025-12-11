import os
import sys
import time
import subprocess
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from yacs.config import CfgNode as CN
import matplotlib.pyplot as plt

# =====================================================================================
# --- 1. GLOBAL CONFIGURATION ---
# !!! VERIFY THESE PATHS BEFORE RUNNING !!!
DATA_DIR = '/mnt/HDD10/ydeole/'  # Directory containing 'train', 'val', etc.
HRNET_REPO_PATH = '/mnt/HDD10/ydeole/HRNet-Semantic-Segmentation'

# --- Experiment Settings ---
EXPERIMENTS = [
    {'id': '01_baseline_adam', 'lr': 1e-4, 'optimizer': 'Adam', 'augmentations': 'standard'},
    {'id': '02_lower_lr_adam', 'lr': 5e-5, 'optimizer': 'Adam', 'augmentations': 'standard'},
    {'id': '03_adamw',         'lr': 1e-4, 'optimizer': 'AdamW', 'augmentations': 'standard'},
    {'id': '04_strong_aug',    'lr': 1e-4, 'optimizer': 'AdamW', 'augmentations': 'strong'},
]
EPOCHS = 25
BATCH_SIZE = 40
NUM_WORKERS = 15 # Your optimized number
IMAGE_HEIGHT = 400
IMAGE_WIDTH = 400
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_FILE = 'tuning_log.csv'
PLOT_FILE = 'training_plots.png'
# =====================================================================================

# --- 2. SETUP & IMPORTS ---
print("--- Initializing Setup ---")
HRNET_LIB_PATH = os.path.join(HRNET_REPO_PATH, 'lib')
if HRNET_LIB_PATH not in sys.path:
    sys.path.append(HRNET_LIB_PATH)

try:
    from models import seg_hrnet
    print("Successfully imported HRNet model library.")
except ImportError:
    print(f"FATAL: Could not import HRNet. Check path: {HRNET_LIB_PATH}")
    sys.exit(1)

np.int = np.int_  # NumPy compatibility shim

# --- 3. DATA HANDLING SECTION ---
def get_verified_file_list(image_dir, mask_dir):
    image_files, mask_files = set(os.listdir(image_dir)), set(os.listdir(mask_dir))
    verified_files = sorted(list(image_files.intersection(mask_files)))
    if len(verified_files) < len(image_files):
        print(f"Warning: {len(image_files) - len(verified_files)} images mismatched and skipped.")
    return verified_files

class PVDataset(Dataset):
    def __init__(self, image_dir, mask_dir, file_list, transform=None):
        self.image_dir, self.mask_dir, self.file_list, self.transform = image_dir, mask_dir, file_list, transform
    def __len__(self): return len(self.file_list)
    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path, mask_path = os.path.join(self.image_dir, img_name), os.path.join(self.mask_dir, img_name)
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        return image, mask

# --- 4. MODEL & TRAINING FUNCTIONS ---
def get_hrnet_model():
    PRETRAINED_WEIGHTS_PATH = os.path.join(DATA_DIR, "HRNet_W18_C_ssld_pretrained.pth")
    if not os.path.exists(PRETRAINED_WEIGHTS_PATH):
        print("Downloading pre-trained weights...")
        url = "https://github.com/HRNet/HRNet-Image-Classification/releases/download/PretrainedWeights/HRNet_W18_C_ssld_pretrained.pth"
        subprocess.run(['wget', '-O', PRETRAINED_WEIGHTS_PATH, url])

    config = CN(); config.MODEL = CN(); config.MODEL.PRETRAINED = PRETRAINED_WEIGHTS_PATH; config.MODEL.NAME = 'seg_hrnet'
    config.MODEL.EXTRA = CN(); config.MODEL.EXTRA.FINAL_CONV_KERNEL = 1
    config.MODEL.EXTRA.STAGE1 = CN({'NUM_MODULES': 1, 'NUM_BRANCHES': 1, 'BLOCK': 'BOTTLENECK', 'NUM_BLOCKS': [4], 'NUM_CHANNELS': [64], 'FUSE_METHOD': 'SUM'})
    config.MODEL.EXTRA.STAGE2 = CN({'NUM_MODULES': 1, 'NUM_BRANCHES': 2, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4], 'NUM_CHANNELS': [18, 36], 'FUSE_METHOD': 'SUM'})
    config.MODEL.EXTRA.STAGE3 = CN({'NUM_MODULES': 4, 'NUM_BRANCHES': 3, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4, 4], 'NUM_CHANNELS': [18, 36, 72], 'FUSE_METHOD': 'SUM'})
    config.MODEL.EXTRA.STAGE4 = CN({'NUM_MODULES': 3, 'NUM_BRANCHES': 4, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4, 4, 4], 'NUM_CHANNELS': [18, 36, 72, 144], 'FUSE_METHOD': 'SUM'})
    config.DATASET = CN({'NUM_CLASSES': 1}); config.MODEL.ALIGN_CORNERS = False
    model = seg_hrnet.get_seg_model(config)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model.to(DEVICE)

def train_one_epoch(loader, model, optimizer, loss_fn):
    model.train(); running_loss = 0.0
    for data, targets in tqdm(loader, desc="Training", leave=False):
        data, targets = data.to(DEVICE), targets.float().unsqueeze(1).to(DEVICE)
        predictions = model(data)
        predictions = F.interpolate(predictions, size=targets.shape[2:], mode='bilinear', align_corners=False)
        loss = loss_fn(predictions, targets)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def evaluate_model(loader, model):
    model.eval(); all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validating", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
            preds_logits = model(x)
            preds_logits = F.interpolate(preds_logits, size=y.shape[2:], mode='bilinear', align_corners=False)
            all_preds.append(torch.sigmoid(preds_logits).cpu()); all_targets.append(y.cpu())
    all_preds, all_targets = torch.cat(all_preds), torch.cat(all_targets)
    tp, fp, fn, tn = smp.metrics.get_stats(all_preds, all_targets.long(), mode='binary', threshold=0.5)
    metrics = {
        'iou': smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item(),
        'precision': smp.metrics.precision(tp, fp, fn, tn, reduction="micro").item(),
        'recall': smp.metrics.recall(tp, fp, fn, tn, reduction="micro").item(),
        'f1': smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item(),
    }
    return metrics

# --- 5. MAIN EXECUTION BLOCK ---
def main():
    print(f"--- Starting Automated Hyperparameter Tuning ---")
    print(f"Device: {DEVICE}, GPUs: {torch.cuda.device_count()}")
    
    # --- Define Augmentation Pipelines ---
    augmentations = {
        'standard': A.Compose([A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH), A.HorizontalFlip(p=0.5), A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()]),
        'strong': A.Compose([A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH), A.ShiftScaleRotate(p=0.8), A.RandomBrightnessContrast(p=0.8), A.HueSaturationValue(p=0.5), A.GaussianBlur(p=0.3), A.GaussNoise(p=0.3), A.CoarseDropout(p=0.5), A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()])
    }
    val_transform = A.Compose([A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH), A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()])

    # --- Load Data Once ---
    train_files = get_verified_file_list(os.path.join(DATA_DIR, 'train/img'), os.path.join(DATA_DIR, 'train/mask'))
    val_files = get_verified_file_list(os.path.join(DATA_DIR, 'val/img'), os.path.join(DATA_DIR, 'val/mask'))
    val_dataset = PVDataset(os.path.join(DATA_DIR, 'val/img'), os.path.join(DATA_DIR, 'val/mask'), val_files, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # --- Run Experiments ---
    results_log = []
    overall_best_iou = -1.0
    best_history = None
    
    for exp in EXPERIMENTS:
        print(f"\n{'='*50}\nRunning Experiment: {exp['id']}\n{'='*50}")
        print(f"Params: LR={exp['lr']}, Optimizer={exp['optimizer']}, Augmentations={exp['augmentations']}")
        
        # --- Setup for this specific experiment ---
        train_transform = augmentations[exp['augmentations']]
        train_dataset = PVDataset(os.path.join(DATA_DIR, 'train/img'), os.path.join(DATA_DIR, 'train/mask'), train_files, train_transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        
        model = get_hrnet_model()
        if exp['optimizer'] == 'Adam': optimizer = torch.optim.Adam(model.parameters(), lr=exp['lr'])
        elif exp['optimizer'] == 'AdamW': optimizer = torch.optim.AdamW(model.parameters(), lr=exp['lr'], weight_decay=1e-2)
        
        loss_fn = lambda p, t: nn.BCEWithLogitsLoss()(p, t) + smp.losses.DiceLoss(mode='binary', from_logits=True)(p, t)
        
        exp_best_iou = -1.0
        current_history = {'train_loss': [], 'val_iou': []}

        for epoch in range(EPOCHS):
            start_time = time.time()
            train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn)
            val_metrics = evaluate_model(val_loader, model)
            end_time = time.time()
            
            val_iou = val_metrics['iou']
            current_history['train_loss'].append(train_loss); current_history['val_iou'].append(val_iou)
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f} | Time: {end_time-start_time:.2f}s")

            if val_iou > exp_best_iou: exp_best_iou = val_iou
            if val_iou > overall_best_iou:
                overall_best_iou = val_iou
                best_history = current_history.copy()
                save_path = f"HRNet-Seg_cnn_best_model_{exp['id']}_iou{val_iou:.4f}.pth"
                print(f"*** New overall best IoU! Saving model to {save_path} ***")
                torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), save_path)

        results_log.append({**exp, 'best_val_iou': exp_best_iou})
        pd.DataFrame(results_log).to_csv(LOG_FILE, index=False)
        
    print("\n--- Tuning Complete ---")
    final_results = pd.read_csv(LOG_FILE)
    print("Final Results Summary:")
    print(final_results)
    
    # --- Save plots for the BEST overall model ---
    if best_history:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(best_history['train_loss'], label='Best Model Training Loss', color='blue')
        ax.plot(best_history['val_iou'], label='Best Model Validation IoU', color='green')
        ax.set_title('Best Model Convergence'); ax.set_xlabel('Epochs'); ax.set_ylabel('Value'); ax.grid(True); ax.legend()
        plt.savefig(PLOT_FILE)
        plt.close(fig)
        print(f"\nConvergence plot for the best model saved to {PLOT_FILE}")

if __name__ == '__main__':
    main()
