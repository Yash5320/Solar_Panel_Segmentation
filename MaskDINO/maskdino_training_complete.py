#!/usr/bin/env python3

import os
import sys
import json
import csv
import torch
import numpy as np
import warnings
from torch.cuda.amp import autocast, GradScaler
from torchmetrics import JaccardIndex, Precision, Recall, F1Score, Accuracy
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import cv2
import logging

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['PYTHONWARNINGS'] = 'ignore'

# Add MaskDINO to path
sys.path.append("/mnt/HDD10/shrushti/MaskDINO")

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, Instances
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from maskdino import add_maskdino_config

class Config:

    # Data paths
    TRAIN_IMG_DIR = "/mnt/HDD10/shrushti/train/img"
    TRAIN_MASK_DIR = "/mnt/HDD10/shrushti/train/mask"
    VAL_IMG_DIR = "/mnt/HDD10/shrushti/val/img"
    VAL_MASK_DIR = "/mnt/HDD10/shrushti/val/mask"

    # Model paths
    OUTPUT_DIR = "/mnt/HDD10/shrushti/MasF3kDINO_outputs_seg"
    CFG_FILE = "/mnt/HDD10/shrushti/MaskDINO/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml"
    PRETRAIN = "/mnt/HDD10/shrushti/pretrained/R-50.pkl"

    # Training parameters
    EPOCHS = 100
    BATCH_SIZE = 4
    LR = 1e-4
    WEIGHT_DECAY = 0.0001
    GRAD_CLIP = 1.0

    # Image settings
    IMAGE_SIZE = 400

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    LOG_INTERVAL = 1  # Log every N epochs
    SAVE_INTERVAL = 1  # Save checkpoint every N epochs


def setup_logging(output_dir):

    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("MASK DINO Binary Segmentation Training")
    logger.info("="*80)
    logger.info(f"Log file: {log_file}")

    return logger


def load_data(img_dir, mask_dir, logger):

    data = []
    img_files = sorted(os.listdir(img_dir))

    for img_file in img_files:
        if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(img_dir, img_file)
        mask_path = os.path.join(mask_dir, img_file)

        if not os.path.exists(mask_path):
            logger.warning(f"Missing mask for {img_file}")
            continue

        data.append((img_path, mask_path))

    return data


def register_datasets(train_data, val_data, logger):

    # Clear existing registrations
    for name in ["binary_train", "binary_val"]:
        if name in DatasetCatalog:
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)

    # Register new datasets
    DatasetCatalog.register("binary_train",
        lambda: [{"file_name": x[0], "sem_seg_file_name": x[1]} for x in train_data])
    DatasetCatalog.register("binary_val",
        lambda: [{"file_name": x[0], "sem_seg_file_name": x[1]} for x in val_data])

    MetadataCatalog.get("binary_train").set(thing_classes=["foreground"])
    MetadataCatalog.get("binary_val").set(thing_classes=["foreground"])

    logger.info(f"Datasets registered: {len(train_data)} train | {len(val_data)} val images")


def prepare_batch(data_list, image_size, device):

    batch = []

    for img_path, mask_path in data_list:
        # Load image and mask
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            continue

        # Resize
        image = cv2.resize(image, (image_size, image_size))
        mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

        # Create binary mask
        fg_mask = (mask > 0).astype(np.uint8)

        # Convert to tensor
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()
        mask_tensor = torch.from_numpy(fg_mask).unsqueeze(0).to(torch.uint8)

        # Create instances
        H, W = image_size, image_size
        instances = Instances((H, W))
        instances.gt_classes = torch.tensor([0], dtype=torch.long)
        instances.gt_masks = mask_tensor

        # Dummy bounding box (box costs are disabled)
        bbox_tensor = torch.tensor([[0.0, 0.0, float(W), float(H)]], dtype=torch.float32)
        instances.gt_boxes = Boxes(bbox_tensor)

        batch.append({
            "image": image_tensor,
            "instances": instances,
            "height": H,
            "width": W,
        })

    return batch


def setup_model(config, logger):

    cfg = get_cfg()
    add_maskdino_config(cfg)
    cfg.merge_from_file(config.CFG_FILE)

    # Model configuration
    cfg.MODEL.META_ARCHITECTURE = "MaskDINO"
    cfg.MODEL.WEIGHTS = config.PRETRAIN
    cfg.MODEL.DEVICE = config.DEVICE

    # Backbone
    cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.MODEL.RESNETS.DEPTH = 50
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.RESNETS.NORM = "FrozenBN"

    # Binary segmentation (2 classes: background + foreground)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2
    cfg.MODEL.MaskDINO.NUM_CLASSES = 2
    cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]

    # Query and test settings
    cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES = 100
    cfg.MODEL.MaskDINO.TEST.INSTANCE_ON = True
    cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON = False
    cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON = False

    # DISABLE BOX LOSSES - only use mask losses for segmentation
    cfg.MODEL.MaskDINO.BOX_WEIGHT = 0.0
    cfg.MODEL.MaskDINO.GIOU_WEIGHT = 0.0
    cfg.MODEL.MaskDINO.COST_BOX = 0.0
    cfg.MODEL.MaskDINO.COST_GIOU = 0.0
    cfg.MODEL.MaskDINO.COST_CLASS = 2.0
    cfg.MODEL.MaskDINO.COST_DICE = 5.0
    cfg.MODEL.MaskDINO.COST_MASK = 5.0

    # Input settings
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]
    cfg.INPUT.FORMAT = "BGR"

    # Dataset
    cfg.DATASETS.TRAIN = ("binary_train",)
    cfg.DATASETS.TEST = ("binary_val",)

    # Build model
    logger.info("Building model...")
    model = build_model(cfg)
    model.to(config.DEVICE)

    # Disable box costs in matcher
    if hasattr(model, 'criterion') and hasattr(model.criterion, 'matcher'):
        if hasattr(model.criterion.matcher, 'cost_bbox'):
            model.criterion.matcher.cost_bbox = 0.0
        if hasattr(model.criterion.matcher, 'cost_giou'):
            model.criterion.matcher.cost_giou = 0.0
        logger.info("Box costs disabled in matcher")

    # Load pretrained weights
    if os.path.exists(config.PRETRAIN):
        logger.info(f"Loading pretrained weights: {config.PRETRAIN}")
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(config.PRETRAIN)
    else:
        logger.warning("No pretrained weights found, training from scratch")

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    return model, cfg


def setup_metrics(device):

    metrics = {
        "iou": JaccardIndex(task="binary").to(device),
        "accuracy": Accuracy(task="binary").to(device),
        "precision": Precision(task="binary").to(device),
        "recall": Recall(task="binary").to(device),
        "f1": F1Score(task="binary").to(device),
    }
    return metrics



def train_epoch(model, optimizer, scaler, data, config, epoch, logger):

    model.train()

    total_loss = 0
    loss_components = defaultdict(float)
    num_batches = 0

    # Shuffle data
    indices = np.random.permutation(len(data))

    pbar = tqdm(range(0, len(data), config.BATCH_SIZE),
                desc=f"Epoch {epoch}/{config.EPOCHS} [Train]",
                leave=True,
                ncols=120)

    for i in pbar:
        batch_indices = indices[i:i+config.BATCH_SIZE]
        batch_data = [data[idx] for idx in batch_indices]

        # Prepare batch
        batch = prepare_batch(batch_data, config.IMAGE_SIZE, config.DEVICE)
        if not batch:
            continue

        batch = [{k: v.to(config.DEVICE) if isinstance(v, torch.Tensor) else v
                 for k, v in b.items()} for b in batch]

        # Forward pass
        optimizer.zero_grad()

        with autocast():
            loss_dict = model(batch)
            losses = sum(loss_dict.values())

        # Backward pass
        scaler.scale(losses).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        # Accumulate losses
        total_loss += losses.item()
        for k, v in loss_dict.items():
            loss_components[k] += v.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({"loss": f"{total_loss/num_batches:.4f}"})

    # Average losses
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_components = {k: v/num_batches for k, v in loss_components.items()} if num_batches > 0 else {}

    return avg_loss, avg_components


@torch.no_grad()
def validate(model, data, metrics, config, epoch, logger):

    model.eval()

    # Reset all metrics
    for m in metrics.values():
        m.reset()

    val_loss = 0
    num_batches = 0

    pbar = tqdm(range(0, len(data), config.BATCH_SIZE),
                desc=f"Epoch {epoch}/{config.EPOCHS} [Val]",
                leave=False,
                ncols=120)

    for i in pbar:
        batch_data = data[i:i+config.BATCH_SIZE]
        batch = prepare_batch(batch_data, config.IMAGE_SIZE, config.DEVICE)
        if not batch:
            continue

        batch = [{k: v.to(config.DEVICE) if isinstance(v, torch.Tensor) else v
                 for k, v in b.items()} for b in batch]

        # Get predictions
        outputs = model(batch)

        # Process each sample
        for data_item, output in zip(batch, outputs):
            # Ground truth
            gt_mask = data_item["instances"].gt_masks.to(config.DEVICE)
            gt = (gt_mask.sum(0) > 0).float()

            # Prediction
            if len(output["instances"]) > 0:
                pred_masks = output["instances"].pred_masks
                pred = (pred_masks.sum(0) > 0.5).float()
            else:
                pred = torch.zeros_like(gt)

            # Update metrics
            for m in metrics.values():
                m.update(pred.unsqueeze(0), gt.unsqueeze(0).int())

    # Compute final metrics
    results = {name: m.compute().item() for name, m in metrics.items()}

    return results


def save_checkpoint(model, optimizer, scaler, epoch, best_iou, history, config, is_best=False):

    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_iou": best_iou,
        "history": dict(history),
        "config": {
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "lr": config.LR,
            "image_size": config.IMAGE_SIZE,
        }
    }

    # Save latest checkpoint
    ckpt_dir = Path(config.OUTPUT_DIR) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    latest_path = ckpt_dir / "checkpoint_latest.pth"
    torch.save(checkpoint, latest_path)

    # Save epoch checkpoint
    epoch_path = ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pth"
    torch.save(checkpoint, epoch_path)

    # Save best checkpoint
    if is_best:
        best_path = ckpt_dir / "checkpoint_best.pth"
        torch.save(checkpoint, best_path)

    return str(latest_path)


def save_metrics_to_json(history, output_dir):

    metrics_dir = Path(output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    json_path = metrics_dir / "metrics_history.json"

    with open(json_path, 'w') as f:
        json.dump(dict(history), f, indent=4)

    return str(json_path)


def save_metrics_to_csv(history, output_dir):

    metrics_dir = Path(output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    csv_path = metrics_dir / "training_metrics.csv"

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'Epoch', 'Train_Loss', 'Val_Loss', 'Val_IoU',
            'Val_Accuracy', 'Val_Precision', 'Val_Recall', 'Val_F1'
        ])

        # Data
        num_epochs = len(history['train_loss'])
        for i in range(num_epochs):
            writer.writerow([
                i + 1,
                f"{history['train_loss'][i]:.6f}",
                f"{history.get('val_loss', [0]*num_epochs)[i]:.6f}",
                f"{history['val_iou'][i]:.6f}",
                f"{history['val_accuracy'][i]:.6f}",
                f"{history['val_precision'][i]:.6f}",
                f"{history['val_recall'][i]:.6f}",
                f"{history['val_f1'][i]:.6f}"
            ])

    return str(csv_path)


def resume_from_checkpoint(checkpoint_path, model, optimizer, scaler, config, logger):

    if not os.path.exists(checkpoint_path):
        logger.info("No checkpoint found, starting from scratch")
        return 1, 0.0, defaultdict(list)

    logger.info(f"Resuming from checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])

    start_epoch = checkpoint["epoch"] + 1
    best_iou = checkpoint.get("best_iou", 0.0)
    history = checkpoint.get("history", defaultdict(list))

    # Convert to defaultdict if needed
    if not isinstance(history, defaultdict):
        history = defaultdict(list, history)

    logger.info(f"Resumed from epoch {checkpoint['epoch']}")
    logger.info(f"Best IoU so far: {best_iou:.4f}")

    return start_epoch, best_iou, history


def main():


    # Setup
    config = Config()
    logger = setup_logging(config.OUTPUT_DIR)

    # Device info
    logger.info(f"Device: {config.DEVICE}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Load data
    logger.info("\nLoading datasets...")
    train_data = load_data(config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR, logger)
    val_data = load_data(config.VAL_IMG_DIR, config.VAL_MASK_DIR, logger)

    if not train_data or not val_data:
        logger.error("No data found! Please check your data directories.")
        return

    # Register datasets
    register_datasets(train_data, val_data, logger)

    # Setup model
    model, cfg = setup_model(config, logger)

    # Setup optimizer and scaler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )
    scaler = GradScaler()

    # Setup metrics
    metrics = setup_metrics(config.DEVICE)
    logger.info(f"Metrics: {list(metrics.keys())}")

    # Resume from checkpoint if exists
    checkpoint_path = Path(config.OUTPUT_DIR) / "checkpoints" / "checkpoint_latest.pth"
    start_epoch, best_iou, history = resume_from_checkpoint(
        checkpoint_path, model, optimizer, scaler, config, logger
    )

    # Training loop
    logger.info("\n" + "="*80)
    logger.info(f"Starting training from epoch {start_epoch} to {config.EPOCHS}")
    logger.info("Press Ctrl+C to stop training early")
    logger.info("="*80 + "\n")

    try:
        for epoch in range(start_epoch, config.EPOCHS + 1):
            epoch_start = datetime.now()

            # Train
            train_loss, loss_components = train_epoch(
                model, optimizer, scaler, train_data, config, epoch, logger
            )

            # Validate
            val_metrics = validate(
                model, val_data, metrics, config, epoch, logger
            )

            # Store metrics
            history["train_loss"].append(train_loss)
            history["val_iou"].append(val_metrics["iou"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_precision"].append(val_metrics["precision"])
            history["val_recall"].append(val_metrics["recall"])
            history["val_f1"].append(val_metrics["f1"])

            # Check if best model
            is_best = val_metrics["iou"] > best_iou
            if is_best:
                best_iou = val_metrics["iou"]

            # Log epoch summary
            epoch_time = (datetime.now() - epoch_start).total_seconds()
            logger.info(f"\nEpoch {epoch}/{config.EPOCHS} Summary:")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Val IoU: {val_metrics['iou']:.4f}")
            logger.info(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"  Val Precision: {val_metrics['precision']:.4f}")
            logger.info(f"  Val Recall: {val_metrics['recall']:.4f}")
            logger.info(f"  Val F1: {val_metrics['f1']:.4f}")
            if is_best:
                logger.info(f"  *** NEW BEST MODEL (IoU: {best_iou:.4f}) ***")
            logger.info(f"  Epoch Time: {epoch_time:.2f}s")
            logger.info("-"*80)

            # Save checkpoint
            if epoch % config.SAVE_INTERVAL == 0:
                save_checkpoint(
                    model, optimizer, scaler, epoch, best_iou, history, config, is_best
                )

            # Save metrics
            save_metrics_to_json(history, config.OUTPUT_DIR)
            save_metrics_to_csv(history, config.OUTPUT_DIR)

    except KeyboardInterrupt:
        logger.info("\n" + "="*80)
        logger.info("Training interrupted by user!")
        logger.info("="*80)

        # Save final checkpoint
        save_checkpoint(
            model, optimizer, scaler, epoch, best_iou, history, config, False
        )

        # Save final metrics
        json_path = save_metrics_to_json(history, config.OUTPUT_DIR)
        csv_path = save_metrics_to_csv(history, config.OUTPUT_DIR)

        logger.info(f"Stopped at epoch {epoch}")
        logger.info(f"Best IoU: {best_iou:.4f}")
        logger.info(f"Metrics saved to: {csv_path}")
        logger.info(f"Checkpoints saved to: {config.OUTPUT_DIR}/checkpoints")
        logger.info("="*80)

        return

    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)

        # Save emergency checkpoint
        save_checkpoint(
            model, optimizer, scaler, epoch, best_iou, history, config, False
        )
        save_metrics_to_json(history, config.OUTPUT_DIR)
        save_metrics_to_csv(history, config.OUTPUT_DIR)

        raise

    # Training completed
    logger.info("\n" + "="*80)
    logger.info("Training Completed Successfully!")
    logger.info(f"Best IoU: {best_iou:.4f}")
    logger.info(f"Total Epochs: {config.EPOCHS}")
    logger.info(f"Checkpoints: {config.OUTPUT_DIR}/checkpoints")
    logger.info(f"Metrics: {config.OUTPUT_DIR}/metrics")
    logger.info("="*80)


if __name__ == "__main__":
    main()
