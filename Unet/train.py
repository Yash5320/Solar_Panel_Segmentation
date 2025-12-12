import os, argparse, random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from dataset import PVSegDataset
from metrics import seg_metrics
from utils import count_params, save_best
from models.unet import UNet
from models.swin_unet import SwinUNet
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model(name: str):
    name = name.lower()
    if name == "unet":
        return UNet(in_ch=3, out_ch=1, base=64)
    elif name == "swin-unet":
        return SwinUNet(backbone="swin_tiny_patch4_window7_224", out_ch=1)
    else:
        raise ValueError(f"Unknown model: {name}")

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, eps: float = 1e-6):
        probs = torch.sigmoid(logits)
        num = 2.0 * (probs * targets).sum(dim=(1, 2, 3))
        den = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps
        dice = num / den
        return 1.0 - dice.mean()

def main(args):
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=args.logdir)

    meta = pd.read_csv(os.path.join(args.root, "metadata.csv"))
    g_df = meta.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n = len(g_df)
    n_train = int(0.8 * n)
    train_df, val_df = g_df.iloc[:n_train], g_df.iloc[n_train:]

    train_ds = PVSegDataset(args.root, domain="google", df=train_df, split="train", aug=True)
    val_ds   = PVSegDataset(args.root, domain="google", df=val_df,   split="val",   aug=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = get_model(args.model).to(device)
    total, trainable = count_params(model)
    print(f"Total params: {total/1e6:.2f}M | Trainable: {trainable/1e6:.2f}M")
    writer.add_text("params", f"total={total}, trainable={trainable}")

    pos_weight = torch.tensor([args.pos_weight], device=device)
    bce  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    dice = DiceLoss()

    def criterion(logits, msks):
        return bce(logits, msks) + 0.5 * dice(logits, msks)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda')

    best_iou, best_path = 0.0, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        for imgs, msks, _ in pbar:
            imgs, msks = imgs.to(device), msks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=args.amp):
                logits = model(imgs)
                loss = criterion(logits, msks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tr_loss += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=float(loss.item()))
        tr_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        cached_logits, cached_targets = [], []
        with torch.no_grad():
            for imgs, msks, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                imgs, msks = imgs.to(device), msks.to(device)
                logits = model(imgs)
                loss = criterion(logits, msks)
                val_loss += loss.item() * imgs.size(0)

                cached_logits.append(logits.detach().float().cpu())
                cached_targets.append(msks.detach().float().cpu())
        val_loss /= len(val_loader.dataset)

        logits_all  = torch.cat(cached_logits,  dim=0)
        targets_all = torch.cat(cached_targets, dim=0)

        best = {"thr": 0.5, "iou": -1.0, "f1": -1.0, "precision": 0.0, "recall": 0.0}
        thr_values = np.arange(args.thr_min, args.thr_max + 1e-9, args.thr_step)
        for thr in thr_values:
            m = seg_metrics(logits_all, targets_all, threshold=float(thr))
            if m[args.select_metric] > best[args.select_metric]:
                best = {"thr": float(thr), **m}

        miou = best["iou"]

        writer.add_scalar("loss/train", tr_loss, epoch)
        writer.add_scalar("loss/val",   val_loss, epoch)
        writer.add_scalar("metrics/best_threshold", best["thr"], epoch)
        writer.add_scalar("metrics/iou",      best["iou"], epoch)
        writer.add_scalar("metrics/f1",       best["f1"], epoch)
        writer.add_scalar("metrics/precision",best["precision"], epoch)
        writer.add_scalar("metrics/recall",   best["recall"], epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        print(
            f"[Val] thr={best['thr']:.2f} loss={val_loss:.4f} "
            f"IoU={best['iou']:.4f} F1={best['f1']:.4f} "
            f"P={best['precision']:.4f} R={best['recall']:.4f}"
        )

        selected_value = best[args.select_metric]
        if selected_value > best_iou:
            best_iou = selected_value
            best_path = save_best(model, epoch, best_iou, args.outdir, args.model.lower())
            print(f"\u2713 New best saved to: {best_path}")

        scheduler.step()

    print(f"Best {args.select_metric}={best_iou:.4f} | checkpoint={best_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="../")
    ap.add_argument("--model", type=str, choices=["unet", "swin-unet"], required=True)

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--pos_weight", type=float, default=5.0,
                    help="Positive class weight for BCEWithLogitsLoss (tune 2–10)")

    ap.add_argument("--outdir", type=str, default="./checkpoints")
    ap.add_argument("--logdir", type=str, default="./runs")

    ap.add_argument("--thr_min", type=float, default=0.10)
    ap.add_argument("--thr_max", type=float, default=0.85)
    ap.add_argument("--thr_step", type=float, default=0.05)
    ap.add_argument("--select_metric", type=str, default="iou", choices=["iou", "f1"],
                    help="Which metric to maximize for model selection")

    args = ap.parse_args()
    main(args)
