# train_final_model.py
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
# --- 1. GLOBAL CONFIGURATION FOR THE FINAL MODEL ---
DATA_DIR = '/mnt/HDD10/ydeole/'
HRNET_REPO_PATH = '/mnt/HDD10/ydeole/HRNet-Semantic-Segmentation'

# --- Champion Hyperparameters ---
LEARNING_RATE = 1e-4
OPTIMIZER_TYPE = 'AdamW'
AUGMENTATION_TYPE = 'strong'

# --- Increased Epochs for Final Training ---
EPOCHS = 50 # Increased for the final run

BATCH_SIZE = 40
NUM_WORKERS = 20
IMAGE_HEIGHT = 400
IMAGE_WIDTH = 400
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =====================================================================================

# (The rest of the script contains the same helper functions as before)
# ... [You can copy-paste the sections for Setup, Data Handling, and Model/Training Functions from the previous script] ...
# --- 2. SETUP & IMPORTS ---
print("--- Initializing Setup ---")
HRNET_LIB_PATH = os.path.join(HRNET_REPO_PATH, 'lib')
if HRNET_LIB_PATH not in sys.path: sys.path.append(HRNET_LIB_PATH)
try: from models import seg_hrnet; print("Successfully imported HRNet.")
except ImportError: print(f"FATAL: Could not import HRNet."); sys.exit(1)
np.int = np.int_

# --- 3. DATA HANDLING SECTION ---
def get_verified_file_list(image_dir, mask_dir):
    image_files, mask_files = set(os.listdir(image_dir)), set(os.listdir(mask_dir))
    return sorted(list(image_files.intersection(mask_files)))

class PVDataset(Dataset):
    def __init__(self, image_dir, mask_dir, file_list, transform=None):
        self.image_dir, self.mask_dir, self.file_list, self.transform = image_dir, mask_dir, file_list, transform
    def __len__(self): return len(self.file_list)
    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path, mask_path = os.path.join(self.image_dir, img_name), os.path.join(self.mask_dir, img_name)
        image, mask = np.array(Image.open(img_path).convert("RGB")), np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0
        if self.transform: image, mask = self.transform(image=image, mask=mask).values()
        return image, mask

# --- 4. MODEL & TRAINING FUNCTIONS ---
def get_hrnet_model():
    PRETRAINED_WEIGHTS_PATH = os.path.join(DATA_DIR, "HRNet_W18_C_ssld_pretrained.pth")
    config = CN(); config.MODEL=CN(); config.MODEL.PRETRAINED=PRETRAINED_WEIGHTS_PATH; config.MODEL.NAME='seg_hrnet'; config.MODEL.EXTRA=CN(); config.MODEL.EXTRA.FINAL_CONV_KERNEL=1
    config.MODEL.EXTRA.STAGE1=CN({'NUM_MODULES':1,'NUM_BRANCHES':1,'BLOCK':'BOTTLENECK','NUM_BLOCKS':[4],'NUM_CHANNELS':[64],'FUSE_METHOD':'SUM'})
    config.MODEL.EXTRA.STAGE2=CN({'NUM_MODULES':1,'NUM_BRANCHES':2,'BLOCK':'BASIC','NUM_BLOCKS':[4,4],'NUM_CHANNELS':[18,36],'FUSE_METHOD':'SUM'})
    config.MODEL.EXTRA.STAGE3=CN({'NUM_MODULES':4,'NUM_BRANCHES':3,'BLOCK':'BASIC','NUM_BLOCKS':[4,4,4],'NUM_CHANNELS':[18,36,72],'FUSE_METHOD':'SUM'})
    config.MODEL.EXTRA.STAGE4=CN({'NUM_MODULES':3,'NUM_BRANCHES':4,'BLOCK':'BASIC','NUM_BLOCKS':[4,4,4,4],'NUM_CHANNELS':[18,36,72,144],'FUSE_METHOD':'SUM'})
    config.DATASET=CN({'NUM_CLASSES':1}); config.MODEL.ALIGN_CORNERS=False
    model = seg_hrnet.get_seg_model(config)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
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
            preds_logits = model(x); preds_logits = F.interpolate(preds_logits, size=y.shape[2:], mode='bilinear', align_corners=False)
            all_preds.append(torch.sigmoid(preds_logits).cpu()); all_targets.append(y.cpu())
    all_preds, all_targets = torch.cat(all_preds), torch.cat(all_targets)
    tp, fp, fn, tn = smp.metrics.get_stats(all_preds, all_targets.long(), mode='binary', threshold=0.5)
    return smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()

# --- 5. MAIN EXECUTION BLOCK ---
def main():
    print(f"--- Starting FINAL Training Run for Champion Model ---")
    print(f"Device: {DEVICE}, GPUs: {torch.cuda.device_count()}, Epochs: {EPOCHS}")
    
    # --- Setup Champion Config ---
    augmentations = {
        'standard': A.Compose([A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH), A.HorizontalFlip(p=0.5), A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()]),
        'strong': A.Compose([A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH), A.ShiftScaleRotate(p=0.8), A.RandomBrightnessContrast(p=0.8), A.HueSaturationValue(p=0.5), A.GaussianBlur(p=0.3), A.GaussNoise(p=0.3), A.CoarseDropout(p=0.5), A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()])
    }
    train_transform = augmentations[AUGMENTATION_TYPE]
    val_transform = A.Compose([A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH), A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()])

    # --- Load Data ---
    train_files = get_verified_file_list(os.path.join(DATA_DIR, 'train/img'), os.path.join(DATA_DIR, 'train/mask'))
    val_files = get_verified_file_list(os.path.join(DATA_DIR, 'val/img'), os.path.join(DATA_DIR, 'val/mask'))
    train_dataset = PVDataset(os.path.join(DATA_DIR, 'train/img'), os.path.join(DATA_DIR, 'train/mask'), train_files, train_transform)
    val_dataset = PVDataset(os.path.join(DATA_DIR, 'val/img'), os.path.join(DATA_DIR, 'val/mask'), val_files, val_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # --- Run Training ---
    model = get_hrnet_model()
    if OPTIMIZER_TYPE == 'Adam': optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif OPTIMIZER_TYPE == 'AdamW': optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    loss_fn = lambda p, t: nn.BCEWithLogitsLoss()(p, t) + smp.losses.DiceLoss(mode='binary', from_logits=True)(p, t)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1, verbose=True) # Increased patience
    
    best_iou = -1.0
    history = {'train_loss': [], 'val_iou': []}
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn)
        val_iou = evaluate_model(val_loader, model)
        end_time = time.time()
        
        history['train_loss'].append(train_loss); history['val_iou'].append(val_iou)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f} | Time: {end_time-start_time:.2f}s")
        scheduler.step(train_loss)

        if val_iou > best_iou:
            best_iou = val_iou
            save_path = f"HRNet-Seg_FINAL_MODEL_epoch{epoch+1}_iou{val_iou:.4f}.pth"
            print(f"*** New best IoU! Saving model to {save_path} ***")
            torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), save_path)
            
    print(f"\n--- Final Training Complete ---")
    print(f"Final best Validation IoU: {best_iou:.4f}")

    # --- Save Final Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history['train_loss'], label='Training Loss', color='blue')
    ax.plot(history['val_iou'], label='Validation IoU', color='green')
    ax.set_title('Final Model Convergence'); ax.set_xlabel('Epochs'); ax.set_ylabel('Value'); ax.grid(True); ax.legend()
    plt.savefig('final_model_plot.png')
    print(f"Final convergence plot saved to final_model_plot.png")

if __name__ == '__main__':
    main()
