# -*- coding: utf-8 -*-
"""SegFormer-MiT-B2 Training Script

Standalone training script for SegFormer-MiT-B2 model.
"""

import torch, os, random, numpy as np
print("Torch:", torch.__version__)
torch.backends.cudnn.benchmark = True  # Speeds up when input dimensions are fixed
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(42)

# Install required packages (uncomment if needed)
# import subprocess
# subprocess.run(["pip", "-q", "install", "segmentation-models-pytorch==0.3.3", "torchmetrics==1.4.0", "albumentations==1.4.4", "opencv-python==4.10.0.84", "transformers"])

import os, re, shutil, subprocess
from pathlib import Path
import time, json
import cv2, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import platform, psutil

ARC = Path("/content/model_development_dataset.zip")   # Fixed path where you uploaded the file
DST = Path("/content/datasets/model_dev")
shutil.rmtree(DST, ignore_errors=True)
DST.mkdir(parents=True, exist_ok=True)

assert ARC.exists(), f"File not found: {ARC}, please confirm the file name and path"

def run(cmd, check=True, tail=4000):
    print("→", " ".join(map(str, cmd)))
    res = subprocess.run(cmd, text=True, capture_output=True)
    if res.stdout: print(res.stdout[-tail:])
    if res.returncode != 0:
        if res.stderr: print(res.stderr[-tail:])
        if check: raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}")
    return res

# If .z01 multi-volume files exist alongside zip, prioritize 7z
z01 = ARC.with_suffix(".z01")
if z01.exists():
    print("Detected multi-volume .z01 set")
    try:
        subprocess.run(["7z"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("Installing p7zip-full ...")
        run(["apt-get","update","-y"])
        run(["apt-get","install","-y","p7zip-full"])
    run(["7z","x","-y",str(z01), f"-o{str(DST)}"])
else:
    print("Single zip detected")
    run(["unzip","-q",str(ARC), "-d", str(DST)])

print("After extract, top-level:")
for p in sorted(DST.iterdir()):
    print("  ", p)

def find_dataset_root(base_dir: str|Path):
    base = Path(base_dir)
    patterns = [
        ("train/img","val/img","train/mask","val/mask"),
        ("train/images","val/images","train/masks","val/masks"),
        ("images/train","images/val","masks/train","masks/val"),
        ("img/train","img/val","mask/train","mask/val"),
        ("train/images","val/images","train/labels","val/labels"),
        ("train/images","val/images","train/annotation","val/annotation"),
        ("train/images","val/images","train/annotations","val/annotations"),
        ("images/train","images/val","labels/train","labels/val"),
    ]
    for root, dirs, files in os.walk(base):
        rp = Path(root)
        for a,b,c,d in patterns:
            if (rp/a).exists() and (rp/b).exists() and (rp/c).exists() and (rp/d).exists():
                return str(rp)
    return None

DATA_ROOT = find_dataset_root(DST)
assert DATA_ROOT is not None, f"Could not find train/val structure. Please check folder naming in {DST}"
print("✅ DATA_ROOT:", DATA_ROOT)

class PVSegDataset(Dataset):
    def __init__(self, img_dir, msk_dir, transform=None, id_whitelist=None,
                 mask_suffixes=("", "_mask", "_label", "-label", "_gt")):
        self.img_dir = Path(img_dir); self.msk_dir = Path(msk_dir)
        self.transform = transform
        self.id_whitelist = set(id_whitelist) if id_whitelist else None
        self.mask_suffixes = mask_suffixes
        exts = [".png",".jpg",".jpeg",".tif",".tiff",".bmp"]
        imgs = sorted([p for p in self.img_dir.glob("*") if p.suffix.lower() in exts])
        self.samples = []
        for ip in imgs:
            stem = ip.stem
            if self.id_whitelist and stem not in self.id_whitelist:
                continue
            found = None
            for suf in self.mask_suffixes:
                for ext in exts:
                    cand = self.msk_dir / f"{stem}{suf}{ext}"
                    if cand.exists(): found = cand; break
                if found is not None: break
            if found is not None:
                self.samples.append((ip, found))
        if not self.samples:
            raise RuntimeError(f"No image-mask pairs in {img_dir} and {msk_dir}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        ip, mp = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(str(ip), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        if msk.ndim == 3: msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
        msk = (msk > 127).astype(np.uint8)  # Foreground=1, Background=0
        if self.transform:
            out = self.transform(image=img, mask=msk)
            img, msk = out["image"], out["mask"].long()
        else:
            img = ToTensorV2()(image=img)["image"]; msk = torch.from_numpy(msk).long()
        return img, msk, ip.name

def get_train_transform(size=256):
    return A.Compose([
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

def get_val_transform(size=256):
    return A.Compose([
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

from pathlib import Path
rootp = Path(DATA_ROOT)
candidates = [
    {"train_images": rootp/"images"/"train", "val_images": rootp/"images"/"val",
     "train_masks": rootp/"masks"/"train",  "val_masks": rootp/"masks"/"val"},
    {"train_images": rootp/"train"/"images","val_images": rootp/"val"/"images",
     "train_masks": rootp/"train"/"masks",  "val_masks": rootp/"val"/"masks"},
    {"train_images": rootp/"train"/"img",   "val_images": rootp/"val"/"img",
     "train_masks": rootp/"train"/"mask",   "val_masks": rootp/"val"/"mask"},
    {"train_images": rootp/"img"/"train",   "val_images": rootp/"img"/"val",
     "train_masks": rootp/"mask"/"train",   "val_masks": rootp/"mask"/"val"},
    {"train_images": rootp/"images"/"train","val_images": rootp/"images"/"val",
     "train_masks": rootp/"labels"/"train", "val_masks": rootp/"labels"/"val"},
    {"train_images": rootp/"images"/"train","val_images": rootp/"images"/"val",
     "train_masks": rootp/"annotations"/"train", "val_masks": rootp/"annotations"/"val"},
]
ds_paths = None
for o in candidates:
    if all(Path(p).exists() for p in o.values()):
        ds_paths = o; break
assert ds_paths is not None, "Did not match image and annotation folders, please check naming"
print(ds_paths)

# Key hyperparameters
IMG_SIZE   = 256      # Start with 256 for quick testing and comparison. Can be changed to 400 or 512 later for retraining.
BATCH_SIZE = 12
NUM_WORKERS= 4

train_ds = PVSegDataset(ds_paths["train_images"], ds_paths["train_masks"], transform=get_train_transform(IMG_SIZE))
val_ds   = PVSegDataset(ds_paths["val_images"],   ds_paths["val_masks"],   transform=get_val_transform(IMG_SIZE))
print(f"Pairs -> train: {len(train_ds)} | val: {len(val_ds)}")

loader_kwargs = dict(batch_size=BATCH_SIZE, pin_memory=True)
if NUM_WORKERS > 0:
    loader_kwargs.update(dict(num_workers=NUM_WORKERS, persistent_workers=True, prefetch_factor=2))
train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(model_name: str, num_classes: int = 2):
    model_name = model_name.lower()
    if model_name == "segformer_mit_b2":
        from transformers import SegformerForSemanticSegmentation, SegformerConfig
        id2label = {0:"background", 1:"pv"}
        label2id = {"background":0, "pv":1}
        cfg = SegformerConfig.from_pretrained("nvidia/mit-b2", num_labels=num_classes,
                                              id2label=id2label, label2id=label2id)
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b2", config=cfg, ignore_mismatched_sizes=True)
        head_type = "hf_segformer"
    else:
        raise ValueError("model_name must be 'segformer_mit_b2'")

    model = model.to(device)
    try:
        model = model.to(memory_format=torch.channels_last)
    except:
        pass
    return model, head_type

def forward_logits(model, head_type: str, x: torch.Tensor) -> torch.Tensor:
    if head_type == "hf_segformer":
        out = model(pixel_values=x).logits  # [B, C, h', w']
    else:
        raise ValueError(head_type)
    return out

def up_to_target(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
    return logits

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(total=total, trainable=trainable,
                total_M=total/1e6, trainable_M=trainable/1e6)

def metrics_from_logits(logits: torch.Tensor, target: torch.Tensor, eps=1e-7) -> Dict[str,float]:
    # Binary classification, channel dim C=2, foreground=1
    logits = up_to_target(logits, target)
    if logits.shape[1] == 1:
        pred = (torch.sigmoid(logits[:,0]) > 0.5).long()
        tgt  = (target>0).long()
    else:
        pred = torch.argmax(logits, dim=1)
        tgt  = (target==1).long()
    # Calculate tp/fp/fn/tn for PV class
    tp = ((pred==1) & (tgt==1)).sum().item()
    fp = ((pred==1) & (tgt==0)).sum().item()
    fn = ((pred==0) & (tgt==1)).sum().item()
    tn = ((pred==0) & (tgt==0)).sum().item()

    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2*precision*recall / (precision + recall + eps)

    # IoU only for PV class
    inter = tp
    union = tp + fp + fn
    iou   = inter / (union + eps)
    return dict(precision=precision, recall=recall, f1=f1, iou=iou)

class CE_Dice_Loss(nn.Module):
    def __init__(self, ce_weight=0.5):
        super().__init__()
        self.ce_weight = ce_weight
    def forward(self, logits, target):
        logits = up_to_target(logits, target)
        if logits.shape[1] == 1:
            ce = F.binary_cross_entropy_with_logits(logits[:,0], (target>0).float())
            prob = torch.sigmoid(logits[:,0])
            tgt  = (target>0).float()
        else:
            ce = F.cross_entropy(logits, target)
            prob = torch.softmax(logits, dim=1)[:,1]
            tgt  = (target==1).float()
        inter = (prob*tgt).sum(dim=(1,2))
        union = prob.sum(dim=(1,2)) + tgt.sum(dim=(1,2))
        dice  = (2*inter + 1e-7)/(union + 1e-7)
        loss  = self.ce_weight*ce + (1-self.ce_weight)*(1-dice.mean())
        return loss

def train_one_epoch(loader, model, head_type, optimizer, scaler, criterion):
    model.train()
    tot_loss=0.0; tot_m=dict(iou=0,precision=0,recall=0,f1=0); n=0
    for imgs, msks, _ in loader:
        imgs = imgs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        msks = msks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(True):
            logits = forward_logits(model, head_type, imgs)
            loss   = criterion(logits, msks)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        torch.cuda.synchronize()
        with torch.no_grad():
            m = metrics_from_logits(logits, msks)
            bs = imgs.size(0)
            tot_loss += loss.item()*bs
            for k in tot_m: tot_m[k] += m[k]*bs
            n += bs
    out = {k:tot_m[k]/n for k in tot_m}
    out["loss"] = tot_loss/n
    return out

@torch.no_grad()
def eval_one_epoch(loader, model, head_type, criterion):
    model.eval()
    tot_loss=0.0; tot_m=dict(iou=0,precision=0,recall=0,f1=0); n=0
    for imgs, msks, _ in loader:
        imgs = imgs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        msks = msks.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(True):
            logits = forward_logits(model, head_type, imgs)
            loss   = criterion(logits, msks)
        m = metrics_from_logits(logits, msks)
        bs = imgs.size(0)
        tot_loss += loss.item()*bs
        for k in tot_m: tot_m[k] += m[k]*bs
        n += bs
    out = {k:tot_m[k]/n for k in tot_m}
    out["loss"] = tot_loss/n
    return out

def plot_curves(history, save_path):
    # history: list of dicts with keys ['epoch','train']['loss'] and ['val']['loss']
    epochs = [h["epoch"] for h in history]
    tr = [h["train"]["loss"] for h in history]
    vl = [h["val"]["loss"]   for h in history]
    plt.figure(figsize=(6,4))
    plt.plot(epochs, tr, label="train loss")
    plt.plot(epochs, vl, label="val loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()

def benchmark_infer_400(model, head_type, repeats=50, warmup=10):
    model.eval()
    x = torch.randn(1,3,400,400, device=device).to(memory_format=torch.channels_last)
    # 预热
    with torch.no_grad(), torch.cuda.amp.autocast(True):
        for _ in range(warmup):
            _ = forward_logits(model, head_type, x)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad(), torch.cuda.amp.autocast(True):
        for _ in range(repeats):
            _ = forward_logits(model, head_type, x)
    torch.cuda.synchronize()
    dt = (time.time() - t0)/repeats
    return dt*1000.0  # ms/img

# Computer configuration collection
try:
    from transformers import __version__ as transformers_version
except Exception:
    transformers_version = "n/a"

def get_hw_sw_config():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb  = int(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024)
    else:
        gpu_name = "CPU"
        vram_mb  = 0

    cpu_name = platform.processor() or platform.uname().processor or "unknown"

    return dict(
        gpu_model=gpu_name,
        gpu_mem_mb=vram_mb,
        cpu=cpu_name,
        cpu_cores=psutil.cpu_count(logical=True),
        ram_gb=round(psutil.virtual_memory().total/1024/1024/1024, 2),
        os=platform.platform(),
        cuda=getattr(torch.version, "cuda", "n/a"),
        pytorch=torch.__version__,
        transformers=transformers_version,
    )

HWCFG = get_hw_sw_config()

CHECK_DIR = Path("/content/checkpoints"); CHECK_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "segformer_mit_b2"

def train_and_report(model_name, epochs=80, patience=10, lr=1e-4, weight_decay=1e-4):
    model, head_type = build_model(model_name, num_classes=2)
    total_params = count_params(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    scaler    = torch.cuda.amp.GradScaler(enabled=True)
    criterion = CE_Dice_Loss(0.5)

    best = dict(epoch=0, iou=-1.0, path="")
    history=[]
    no_improve=0
    t_start = time.time()

    for ep in range(1, epochs+1):
        t0 = time.time()
        tr = train_one_epoch(train_loader, model, head_type, optimizer, scaler, criterion)
        vl = eval_one_epoch(val_loader,   model, head_type, criterion)
        scheduler.step()
        dt = time.time()-t0
        history.append(dict(epoch=ep, train=tr, val=vl, minutes=dt/60))
        print(f"[{model_name}][{ep:03d}] "
              f"train loss {tr['loss']:.4f} IoU {tr['iou']:.4f} | "
              f"val loss {vl['loss']:.4f} IoU {vl['iou']:.4f} P {vl['precision']:.4f} R {vl['recall']:.4f} F1 {vl['f1']:.4f} | "
              f"{dt/60:.2f} min")

        if vl["iou"] > best["iou"]:
            best.update(epoch=ep, iou=vl["iou"])
            tmp_path = CHECK_DIR / f"{model_name}_best_tmp.pt"
            torch.save({"model": model.state_dict(),
                        "cfg": dict(model=model_name, img_size=IMG_SIZE),
                        "val": vl}, tmp_path)
            best["path"] = str(tmp_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping")
                break

    train_minutes = (time.time()-t_start)/60
    ms_per_img = benchmark_infer_400(model, head_type, repeats=50, warmup=10)

    curve_path = CHECK_DIR / f"{model_name}_loss_curves.png"
    plot_curves(history, str(curve_path))

    final_name = f"{model_name}_epoch{best['epoch']:03d}_miou{best['iou']:.4f}.pt"
    final_path = CHECK_DIR / final_name
    shutil.move(best["path"], final_path)

    report = dict(
        model=model_name,
        computer_configuration=HWCFG,
        training_minutes=round(train_minutes,2),
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        train_samples=len(train_ds),
        val_samples=len(val_ds),
        loss_curves=str(curve_path),
        inference_time_ms_per_400x400=round(ms_per_img,2),
        parameters=total_params,
        best_epoch=best["epoch"],
        metrics_val_best=dict(
            IoU=history[best["epoch"]-1]["val"]["iou"],
            Precision=history[best["epoch"]-1]["val"]["precision"],
            Recall=history[best["epoch"]-1]["val"]["recall"],
            F1=history[best["epoch"]-1]["val"]["f1"],
            Loss=history[best["epoch"]-1]["val"]["loss"],
        ),
        final_model=str(final_path),
    )
    with open(CHECK_DIR / f"{model_name}_report.json","w") as f:
        json.dump(report, f, indent=2)
    return report

# Train SegFormer-MiT-B2
torch.cuda.empty_cache()
report = train_and_report(MODEL_NAME, epochs=80, patience=10, lr=1e-4, weight_decay=1e-4)

print("Training complete!")
print(f"Best IoU: {report['metrics_val_best']['IoU']:.4f}")
print(f"Model saved: {report['final_model']}")

