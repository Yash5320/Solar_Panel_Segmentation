import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import segmentation_models_pytorch as smp
from yacs.config import CfgNode as CN

# =====================================================================================
# --- 0. NUMPY COMPATIBILITY FIX ---
# =====================================================================================
# The HRNet library uses 'np.int', which was removed in NumPy 1.24+.
# We manually restore it as an alias to the built-in int.
np.int = int 

# =====================================================================================
# --- 1. CONFIGURATION ---
# =====================================================================================

# Current Working Directory: /mnt/HDD10/ydeole/HRNet
CURRENT_DIR = os.getcwd()

# Data Paths (Independent Test)
#TEST_BASE_DIR = '/mnt/HDD10/ydeole/independent_test'
TEST_BASE_DIR = '/mnt/HDD10/ydeole/UCI_Campus_Test'

IMG_DIR = os.path.join(TEST_BASE_DIR, 'img')
MASK_DIR = os.path.join(TEST_BASE_DIR, 'mask')
OUTPUT_DIR = os.path.join(TEST_BASE_DIR, 'HRNetPreds')

# HRNet Repository path
HRNET_REPO_PATH = os.path.join(CURRENT_DIR, 'HRNet-Semantic-Segmentation')

# The Specific Weights File
WEIGHTS_PATH = os.path.join(CURRENT_DIR, 'HRNet-Seg_FINAL_MODEL_epoch49_iou0.9294.pth')

# Parameters
IMAGE_HEIGHT = 400
IMAGE_WIDTH = 400
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1 

# =====================================================================================
# --- 2. SETUP IMPORTS ---
# =====================================================================================
print(f"--- Initializing Environment ---")
HRNET_LIB_PATH = os.path.join(HRNET_REPO_PATH, 'lib')
if HRNET_LIB_PATH not in sys.path:
    sys.path.append(HRNET_LIB_PATH)

try:
    from models import seg_hrnet
    print("Successfully imported HRNet model library.")
except ImportError:
    print(f"FATAL: Could not import HRNet. Check path: {HRNET_LIB_PATH}")
    sys.exit(1)

# =====================================================================================
# --- 3. DATASET CLASS ---
# =====================================================================================
class IndependentTestDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_filename = self.images[idx]
        
        # Logic to find the mask: 400038.png -> 400038_mask.png
        base_name = os.path.splitext(img_filename)[0]
        mask_filename = f"{base_name}_mask.png"
        
        img_path = os.path.join(self.img_dir, img_filename)
        mask_path = os.path.join(self.mask_dir, mask_filename)

        # Load Image
        image = np.array(Image.open(img_path).convert("RGB"))
        original_size = (image.shape[0], image.shape[1]) # H, W

        # Load Mask (for metrics)
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
            mask[mask == 255.0] = 1.0 
        else:
            # Create dummy mask if missing, just to allow script to run
            mask = np.zeros((original_size[0], original_size[1]), dtype=np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image_tensor = augmented['image']
            mask_tensor = augmented['mask']
        
        return image_tensor, mask_tensor, img_filename, original_size

# =====================================================================================
# --- 4. MODEL LOADER ---
# =====================================================================================
def load_hrnet_model(weights_path):
    # Dummy path for config init
    dummy_pretrained = os.path.join(CURRENT_DIR, "HRNet_W18_C_ssld_pretrained.pth")
    
    config = CN()
    config.MODEL = CN()
    config.MODEL.PRETRAINED = dummy_pretrained
    config.MODEL.NAME = 'seg_hrnet'
    config.MODEL.EXTRA = CN()
    config.MODEL.EXTRA.FINAL_CONV_KERNEL = 1
    config.MODEL.EXTRA.STAGE1 = CN({'NUM_MODULES': 1, 'NUM_BRANCHES': 1, 'BLOCK': 'BOTTLENECK', 'NUM_BLOCKS': [4], 'NUM_CHANNELS': [64], 'FUSE_METHOD': 'SUM'})
    config.MODEL.EXTRA.STAGE2 = CN({'NUM_MODULES': 1, 'NUM_BRANCHES': 2, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4], 'NUM_CHANNELS': [18, 36], 'FUSE_METHOD': 'SUM'})
    config.MODEL.EXTRA.STAGE3 = CN({'NUM_MODULES': 4, 'NUM_BRANCHES': 3, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4, 4], 'NUM_CHANNELS': [18, 36, 72], 'FUSE_METHOD': 'SUM'})
    config.MODEL.EXTRA.STAGE4 = CN({'NUM_MODULES': 3, 'NUM_BRANCHES': 4, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4, 4, 4], 'NUM_CHANNELS': [18, 36, 72, 144], 'FUSE_METHOD': 'SUM'})
    config.DATASET = CN({'NUM_CLASSES': 1})
    config.MODEL.ALIGN_CORNERS = False

    model = seg_hrnet.get_seg_model(config)
    
    print(f"Loading weights: {weights_path}")
    state_dict = torch.load(weights_path, map_location=DEVICE)
    
    # Fix 'module.' prefix if the model was saved with DataParallel
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

# =====================================================================================
# --- 5. MAIN EXECUTION ---
# =====================================================================================
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output folder: {OUTPUT_DIR}")

    # Standard validation transform
    test_transform = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2()
    ])

    dataset = IndependentTestDataset(IMG_DIR, MASK_DIR, transform=test_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = load_hrnet_model(WEIGHTS_PATH)
    
    # Stats accumulators
    tp_tot, fp_tot, fn_tot, tn_tot = 0, 0, 0, 0
    
    print(f"--- Processing {len(dataset)} images ---")
    
    with torch.no_grad():
        for image, mask, filename, original_dim in tqdm(loader):
            image = image.to(DEVICE)
            mask = mask.to(DEVICE).unsqueeze(1) # [B, 1, 400, 400]
            filename = filename[0]
            
            # 1. Inference
            logits = model(image)
            
            # 2. Upsample logits to 400x400 (if model output differs) for Metric Calculation
            logits = F.interpolate(logits, size=(IMAGE_HEIGHT, IMAGE_WIDTH), mode='bilinear', align_corners=False)
            probs = torch.sigmoid(logits)
            
            # 3. Calculate Metrics (at 400x400 resolution)
            # We compare the Resized Mask (loaded by dataset) vs Resized Prediction
            tp, fp, fn, tn = smp.metrics.get_stats(probs, mask.long(), mode='binary', threshold=0.5)
            tp_tot += tp; fp_tot += fp; fn_tot += fn; tn_tot += tn
            
            # 4. Save Prediction (Resized back to ORIGINAL resolution)
            pred_mask = (probs > 0.5).float()
            
            # Get original H, W
            orig_h, orig_w = int(original_dim[0]), int(original_dim[1])
            
            # Interpolate 'pred_mask' (1, 1, 400, 400) -> (1, 1, orig_H, orig_W)
            pred_mask_orig = F.interpolate(pred_mask, size=(orig_h, orig_w), mode='nearest')
            
            # Convert to image
            pred_np = pred_mask_orig.squeeze().cpu().numpy() # (H, W)
            pred_img = (pred_np * 255).astype(np.uint8)
            
            Image.fromarray(pred_img).save(os.path.join(OUTPUT_DIR, filename))

    # --- Final Report ---
    iou = smp.metrics.iou_score(tp_tot, fp_tot, fn_tot, tn_tot, reduction="micro").item()
    precision = smp.metrics.precision(tp_tot, fp_tot, fn_tot, tn_tot, reduction="micro").item()
    recall = smp.metrics.recall(tp_tot, fp_tot, fn_tot, tn_tot, reduction="micro").item()
    f1 = smp.metrics.f1_score(tp_tot, fp_tot, fn_tot, tn_tot, reduction="micro").item()
    
    report = (
        f"--- HRNet Inference Report ---\n"
        f"Model: {os.path.basename(WEIGHTS_PATH)}\n"
        f"Images: {IMG_DIR}\n"
        f"Output: {OUTPUT_DIR}\n"
        f"------------------------------\n"
        f"Mean IoU   : {iou:.4f}\n"
        f"Precision  : {precision:.4f}\n"
        f"Recall     : {recall:.4f}\n"
        f"F1 Score   : {f1:.4f}\n"
        f"------------------------------\n"
    )
    
    print(report)
    with open(os.path.join(OUTPUT_DIR, 'metrics_report.txt'), 'w') as f:
        f.write(report)

if __name__ == '__main__':
    main()
