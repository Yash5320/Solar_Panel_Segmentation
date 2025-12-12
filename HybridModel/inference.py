#!/usr/bin/env python3

import os
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Optional imports that may be heavy - handle gracefully
try:
    from mmseg.apis import init_segmentor, inference_segmentor
    MMSeg_AVAILABLE = True
except Exception:
    MMSeg_AVAILABLE = False

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except Exception:
    SMP_AVAILABLE = False

try:
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# HRNet custom repo import - will be dynamically added in loader function
import albumentations as A
from albumentations.pytorch import ToTensorV2

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_image_rgb(path: Path) -> np.ndarray:
    img = np.array(Image.open(path).convert("RGB"))
    return img

def save_mask_png(mask: np.ndarray, path: Path):
    """
    Save a binary mask as PNG.
    mask: binary uint8 array (H, W) with values 0/255 or 0/1
    """
    m = (mask > 0).astype(np.uint8) * 255
    Image.fromarray(m).save(str(path))

def save_overlay(image: np.ndarray, mask_bin: np.ndarray, out_path: Path, alpha: float = 0.5):
    """
    Save an overlay image with the mask in red.
    mask_bin: binary 0/1 array
    """
    overlay = image.copy()
    mask_rgb = np.zeros_like(image)
    mask_rgb[mask_bin == 1] = [255, 0, 0]  # red
    overlay = cv2.addWeighted(overlay, 1 - alpha, mask_rgb, alpha, 0)
    Image.fromarray(overlay).save(str(out_path))

def compute_confusion_and_scores(pred_bin: np.ndarray, gt_bin: np.ndarray) -> Dict[str, float]:
    """
    Compute confusion matrix and standard segmentation metrics.
    pred_bin, gt_bin: binary 0/1 arrays
    Returns: dict with tp, fp, fn, tn, iou, precision, recall, f1
    """
    p = pred_bin.flatten()
    g = gt_bin.flatten()
    tp = int(np.sum((p == 1) & (g == 1)))
    fp = int(np.sum((p == 1) & (g == 0)))
    fn = int(np.sum((p == 0) & (g == 1)))
    tn = int(np.sum((p == 0) & (g == 0)))
    eps = 1e-8
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'iou': iou, 'precision': precision, 'recall': recall, 'f1': f1}

def build_segformer_loader(checkpoint: Optional[str], device: torch.device):
    """
    Initialize SegFormer (MiT-B2) model and return inference function.
    Returns: infer(img: np.ndarray) -> probability map (float32 HxW)
    """
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers not available. pip install transformers and accelerate.")
    print("Loading SegFormer (MiT-B2)...")
    processor = AutoImageProcessor.from_pretrained("nvidia/mit-b2")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2",
        ignore_mismatched_sizes=True
    )
    if checkpoint:
        # Load custom checkpoint if provided
        try:
            sd = torch.load(checkpoint, map_location='cpu')
            if 'model' in sd:
                model.load_state_dict(sd['model'], strict=False)
            else:
                model.load_state_dict(sd, strict=False)
            print(f"Loaded SegFormer checkpoint: {checkpoint}")
        except Exception as e:
            print(f"Warning: could not load SegFormer checkpoint {checkpoint}: {e}")
    model.to(device)
    model.eval()

    def infer(img: np.ndarray) -> np.ndarray:
        H, W = img.shape[:2]
        inputs = processor(images=Image.fromarray(img), return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits
            logits_up = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
            C = logits_up.shape[1]
            if C == 1:
                prob = torch.sigmoid(logits_up[:, 0, :, :])
            else:
                probs = torch.softmax(logits_up, dim=1)
                foreground = 1 if probs.shape[1] > 1 else 0
                prob = probs[:, foreground, :, :]
            prob_np = prob.squeeze().cpu().numpy().astype(np.float32)
            return np.clip(prob_np, 0.0, 1.0)
    return infer

def build_hrnet_loader(hrnet_repo_path: Optional[str], checkpoint: Optional[str], device: torch.device):
    """
    Initialize HRNet segmentation model.
    Returns: infer(img) -> probability map (HxW)
    """
    if hrnet_repo_path is None:
        raise RuntimeError("hrnet_repo_path must be provided to load HRNet model.")
    lib_path = Path(hrnet_repo_path) / "lib"
    if str(lib_path) not in __import__('sys').path:
        __import__('sys').path.append(str(lib_path))

    try:
        from models import seg_hrnet  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Could not import HRNet library from {hrnet_repo_path}: {e}")

    from yacs.config import CfgNode as CN
    config = CN()
    config.MODEL = CN()
    config.MODEL.PRETRAINED = ""
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
    if checkpoint:
        sd = torch.load(checkpoint, map_location='cpu')
        if list(sd.keys())[0].startswith('module.'):
            sd = {k[7:]: v for k, v in sd.items()}
        model.load_state_dict(sd)
    model.to(device)
    model.eval()

    transform = A.Compose([
        A.Resize(400, 400),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2()
    ])

    def infer(img: np.ndarray) -> np.ndarray:
        H, W = img.shape[:2]
        aug = transform(image=img)
        tensor = aug['image'].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            logits_up = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
            if logits_up.shape[1] == 1:
                prob = torch.sigmoid(logits_up[:, 0, :, :])
            else:
                prob = torch.softmax(logits_up, dim=1)[:, 1, :, :]
            prob_np = prob.squeeze().cpu().numpy().astype(np.float32)
            return np.clip(prob_np, 0.0, 1.0)
    return infer

def build_internimage_loader(mmseg_config: Optional[str], mmseg_checkpoint: Optional[str], device: str):
    """
    Initialize InternImage model using mmsegmentation.
    Returns: infer(img) -> probability map (HxW)
    """
    if not MMSeg_AVAILABLE:
        raise RuntimeError("mmseg (OpenMMLab) not available. Install mmsegmentation.")
    if mmseg_config is None or mmseg_checkpoint is None:
        raise RuntimeError("Provide mmseg config and checkpoint paths for InternImage model.")

    print("Loading InternImage (mmseg) segmentor...")
    model = init_segmentor(mmseg_config, mmseg_checkpoint, device=device)
    model.eval()

    def infer(img: np.ndarray) -> np.ndarray:
        """
        Infer probability map from InternImage.
        Returns HxW float32 probability map.
        """
        # Use return_logits=True to extract probabilities instead of label map
        result = inference_segmentor(model, img, return_logits=True)
        logits = result[0]  # [C,H,W]
        if logits.shape[0] == 1:
            prob = torch.sigmoid(torch.from_numpy(logits[0]))
        else:
            prob = torch.softmax(torch.from_numpy(logits), dim=0)[1]
        return np.clip(prob.cpu().numpy().astype(np.float32), 0.0, 1.0)
    return infer

def build_pspnet_loader(psp_checkpoint: Optional[str], device: torch.device):
    """
    Initialize PSPNet using segmentation_models_pytorch.
    Returns: infer(img) -> probability map (HxW)
    """
    if not SMP_AVAILABLE:
        raise RuntimeError("segmentation_models_pytorch (smp) not available. pip install segmentation-models-pytorch")
    encoder = "resnet34"
    print("Building PSPNet (smp) with encoder:", encoder)
    model = smp.PSPNet(
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    )
    if psp_checkpoint:
        try:
            sd = torch.load(psp_checkpoint, map_location='cpu')
            if 'model' in sd:
                sd = sd['model']
            if list(sd.keys())[0].startswith('module.'):
                sd = {k[7:]: v for k, v in sd.items()}
            model.load_state_dict(sd, strict=False)
            print("Loaded PSPNet checkpoint:", psp_checkpoint)
        except Exception as e:
            print("Warning: could not load PSPNet checkpoint:", e)
    model.to(device)
    model.eval()

    transform = A.Compose([
        A.Resize(400, 400),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2()
    ])

    def infer(img: np.ndarray) -> np.ndarray:
        H, W = img.shape[:2]
        aug = transform(image=img)
        tensor = aug['image'].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            logits_up = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
            prob = torch.sigmoid(logits_up[:, 0, :, :])
            return np.clip(prob.squeeze().cpu().numpy().astype(np.float32), 0.0, 1.0)
    return infer

def fuse_prob_maps(prob_maps: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    """
    prob_maps: dict model_name -> (H, W) probabilities in [0,1] for foreground
    weights: dict model_name -> scalar weight (sums to 1 ideally)
    Returns fused foreground probability map shape (H, W)
    """
    names = list(prob_maps.keys())
    assert set(names) == set(weights.keys()), "prob_maps and weights keys must match"
    # get H,W
    H, W = next(iter(prob_maps.values())).shape
    fused = np.zeros((H, W), dtype=np.float32)
    # normalize weights (just in case)
    wsum = sum(weights.values())
    if wsum <= 0:
        raise ValueError("Sum of weights must be > 0")
    for name, prob in prob_maps.items():
        w = weights[name] / wsum
        fused += prob.astype(np.float32) * w
    fused = np.clip(fused, 0.0, 1.0)
    return fused

def parse_args():
    p = argparse.ArgumentParser(description="Hybrid 4-model ensemble inference for rooftop PV segmentation")
    p.add_argument("--img_dir", type=str, required=True, help="Directory with test images (png/jpg)")
    p.add_argument("--mask_dir", type=str, default=None, help="Optional GT masks directory (for metrics). Masks should be *_mask.png or same filename with 255 for fg.")
    p.add_argument("--out_dir", type=str, required=True, help="Directory to save ensemble masks and overlays")
    p.add_argument("--intern_cfg", type=str, default=None, help="mmseg config file for InternImage (optional)")
    p.add_argument("--intern_ckpt", type=str, default=None, help="mmseg checkpoint for InternImage (optional)")
    p.add_argument("--hrnet_repo", type=str, default=None, help="Path to HRNet repo root (contains 'lib' and 'models')")
    p.add_argument("--hrnet_ckpt", type=str, default=None, help="HRNet weights path (optional)")
    p.add_argument("--segformer_ckpt", type=str, default=None, help="SegFormer/custom ckpt path (optional)")
    p.add_argument("--psp_ckpt", type=str, default=None, help="PSPNet weights path (optional)")
    p.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda or cpu)")
    p.add_argument("--save_overlay", action="store_true", help="Save overlay images (red mask on original)")
    p.add_argument("--threshold", type=float, default=0.5, help="Threshold on fused probability to produce final binary mask")
    p.add_argument("--weights", type=str, default=None,
                   help="Optional JSON-like weights override, e.g. '{\"intern\":0.3,\"hrnet\":0.25,\"segformer\":0.25,\"psp\":0.2}'")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    IMG_DIR = Path(args.img_dir)
    MASK_DIR = Path(args.mask_dir) if args.mask_dir else None
    OUT_DIR = Path(args.out_dir)
    ensure_dir(OUT_DIR)
    ensure_dir(OUT_DIR / "masks")
    if args.save_overlay:
        ensure_dir(OUT_DIR / "overlays")
    # default weights based on spec
    weights = {
        "intern": 0.30,
        "hrnet": 0.25,
        "segformer": 0.25,
        "psp": 0.20
    }
    if args.weights:
        import json
        weights = json.loads(args.weights)

    # Build model inference functions
    infer_funcs = {}
    # InternImage (mmseg)
    try:
        if args.intern_cfg and args.intern_ckpt:
            infer_funcs['intern'] = build_internimage_loader(args.intern_cfg, args.intern_ckpt, device=str(device))
            print("InternImage loader ready.")
        else:
            print("Skipping InternImage (mmseg) because config/checkpoint not provided.")
    except Exception as e:
        print("Warning: could not initialize InternImage loader:", e)

    # HRNet
    try:
        if args.hrnet_repo:
            infer_funcs['hrnet'] = build_hrnet_loader(args.hrnet_repo, args.hrnet_ckpt, device)
            print("HRNet loader ready.")
        else:
            print("Skipping HRNet because hrnet_repo not provided.")
    except Exception as e:
        print("Warning: could not initialize HRNet loader:", e)

    # SegFormer
    try:
        infer_funcs['segformer'] = build_segformer_loader(args.segformer_ckpt, device)
        print("SegFormer loader ready.")
    except Exception as e:
        print("Warning: could not initialize SegFormer loader:", e)

    # PSPNet (smp)
    try:
        infer_funcs['psp'] = build_pspnet_loader(args.psp_ckpt, device)
        print("PSPNet loader ready.")
    except Exception as e:
        print("Warning: could not initialize PSPNet loader:", e)

    # Keep only the functions that succeeded
    available_models = list(infer_funcs.keys())
    if not available_models:
        raise RuntimeError("No models available for inference. Provide at least one model's config/checkpoint.")
    print("Available models for ensemble:", available_models)

    # Ensure weights keys match available models: if provided weights include models that are not loaded, ignore them.
    weights = {k: v for k, v in weights.items() if k in available_models}
    # Normalize
    wsum = sum(weights.values())
    if wsum <= 0:
        raise RuntimeError("No positive weights provided for available models")
    weights = {k: float(v) / wsum for k, v in weights.items()}

    # Discover images
    imgs = sorted([p for p in IMG_DIR.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")])
    if not imgs:
        raise RuntimeError("No images found in img_dir")

    # Stats accumulators for global metrics
    global_stats = {'tp':0, 'fp':0, 'fn':0, 'tn':0}
    metrics_list = []

    for img_path in tqdm(imgs, desc="Images"):
        img = read_image_rgb(img_path)
        H, W = img.shape[:2]

        # Per-model probability maps
        prob_maps = {}
        for name, fn in infer_funcs.items():
            try:
                prob = fn(img)  # should return HxW float32 in [0,1]
                # Safety: if shape differs, resize
                if prob.shape != (H, W):
                    prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
                prob_maps[name] = prob.astype(np.float32)
            except Exception as e:
                print(f"Warning: model {name} failed on {img_path.name}: {e}")
                # fallback: zeros
                prob_maps[name] = np.zeros((H, W), dtype=np.float32)

        # Fuse
        fused_prob = fuse_prob_maps(prob_maps, weights)

        # Final binary mask: threshold OR argmax between [1-fused, fused]
        final_bin = (fused_prob >= args.threshold).astype(np.uint8)

        # Save mask and overlay
        mask_save_path = OUT_DIR / "masks" / img_path.name
        save_mask_png(final_bin, mask_save_path)

        if args.save_overlay:
            save_overlay(img, final_bin, OUT_DIR / "overlays" / img_path.name, alpha=0.5)

        # Metrics if GT available
        if MASK_DIR:
            # assume GT mask uses same filename but possibly suffix _mask or same name
            gt_path_candidate1 = MASK_DIR / img_path.name
            gt_path_candidate2 = MASK_DIR / f"{img_path.stem}_mask{img_path.suffix}"
            gt_path = gt_path_candidate1 if gt_path_candidate1.exists() else (gt_path_candidate2 if gt_path_candidate2.exists() else None)
            if gt_path:
                gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                if gt is None:
                    print(f"Warning: couldn't read GT mask: {gt_path}")
                else:
                    gt_bin = (gt > 0).astype(np.uint8)
                    # resize gt if mismatched
                    if gt_bin.shape != final_bin.shape:
                        gt_bin = cv2.resize(gt_bin.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                    stats = compute_confusion_and_scores(final_bin, gt_bin)
                    metrics_list.append({'image': img_path.name, **stats})
                    # accumulate global
                    global_stats['tp'] += stats['tp']
                    global_stats['fp'] += stats['fp']
                    global_stats['fn'] += stats['fn']
                    global_stats['tn'] += stats['tn']

    # Report
    print("\n=== Ensemble Inference Complete ===")
    print(f"Processed {len(imgs)} images. Output saved to {OUT_DIR}")

    if metrics_list:
        # compute global scores
        tp = global_stats['tp']; fp = global_stats['fp']; fn = global_stats['fn']; tn = global_stats['tn']
        eps = 1e-8
        global_iou = tp / (tp + fp + fn + eps)
        global_prec = tp / (tp + fp + eps)
        global_rec = tp / (tp + fn + eps)
        global_f1 = 2 * global_prec * global_rec / (global_prec + global_rec + eps)
        print("\n=== Evaluation (global) ===")
        print(f"TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}")
        print(f"IoU: {global_iou:.4f}")
        print(f"Precision: {global_prec:.4f}")
        print(f"Recall: {global_rec:.4f}")
        print(f"F1: {global_f1:.4f}")

        # Save per-image metrics
        import json
        with open(OUT_DIR / "metrics_per_image.json", "w") as f:
            json.dump(metrics_list, f, indent=2)
        # Save summary
        with open(OUT_DIR / "metrics_summary.txt", "w") as f:
            f.write("Global TP FP FN TN\n")
            f.write(f"{tp} {fp} {fn} {tn}\n")
            f.write(f"IoU {global_iou:.6f}\n")
            f.write(f"Precision {global_prec:.6f}\n")
            f.write(f"Recall {global_rec:.6f}\n")
            f.write(f"F1 {global_f1:.6f}\n")
        print(f"Saved metrics to {OUT_DIR}")

if __name__ == "__main__":
    main()
