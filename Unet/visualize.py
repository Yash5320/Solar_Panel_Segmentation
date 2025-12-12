import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import PVSegDataset
from models.unet import UNet
from models.swin_unet import SwinUNet

DEFAULT_SAMPLE_IDS = [
    "ZLVSC7C2CEKGIW",
    "STFNN6D12IUTJU",
    "IJYJF237BDLZX",
]


def seed_all(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(model_name: str, ckpt_path: str, img_size: int = 400) -> torch.nn.Module:
    model_name = model_name.lower()
    if model_name == "unet":
        model = UNet(in_ch=3, out_ch=1, base=64)
    elif model_name == "swin-unet":
        model = SwinUNet(
            backbone="swin_tiny_patch4_window7_224",
            out_ch=1,
            img_size=img_size,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    return model


def visualize_triplet(img, msk, logit, sample_id: str, outdir: str,
                      threshold: float = 0.5, suffix: str = ""):
    img_np = img.detach().cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np, 0.0, 1.0)

    gt = msk.detach().cpu().numpy()[0]
    prob = torch.sigmoid(logit).detach().cpu().numpy()[0]
    pred = (prob >= threshold).astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(img_np)
    axes[0].set_title("Image")
    axes[1].imshow(gt, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred, cmap="gray")
    axes[2].set_title(f"Prediction (thr={threshold:.2f})")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    fname = f"{sample_id}"
    if suffix:
        fname += f"_{suffix}"
    save_path = os.path.join(outdir, f"{fname}.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] {save_path}")


def build_val_dataset(root: str, seed: int, model_name: str,
                      img_size: int = 400, resize_to: int | None = None):
    meta_path = os.path.join(root, "metadata.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.csv not found under root={root}")

    meta = pd.read_csv(meta_path)
    g_df = meta.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(g_df)
    n_train = int(0.8 * n)
    val_df = g_df.iloc[n_train:]

    normalize = None

    val_ds = PVSegDataset(
        root,
        domain="google",
        df=val_df,
        split="val",
        img_size=img_size,
        aug=False,
        normalize=normalize,
        resize_to=resize_to,
    )
    return val_ds


def main(args):
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.ids is None or len(args.ids) == 0:
        sample_ids = DEFAULT_SAMPLE_IDS
    else:
        sample_ids = args.ids
    sample_ids = list(dict.fromkeys(sample_ids))

    print("Visualizing sample IDs:", ", ".join(sample_ids))

    val_ds = build_val_dataset(
        root=args.root,
        seed=args.seed,
        model_name=args.model,
        img_size=args.img_size,
        resize_to=args.resize_to,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = get_model(args.model, args.ckpt, img_size=args.img_size)
    model.to(device).eval()

    remaining = set(sample_ids)
    found = {}
    with torch.no_grad():
        for imgs, msks, fnames in val_loader:
            fname = fnames[0]
            stem = os.path.splitext(os.path.basename(fname))[0]

            if stem not in remaining:
                continue

            imgs = imgs.to(device)
            msks = msks.to(device)

            logits = model(imgs)

            found[stem] = (
                imgs[0].detach().cpu(),
                msks[0].detach().cpu(),
                logits[0].detach().cpu(),
            )
            remaining.remove(stem)
            print(f"[Found] {stem}")

            if not remaining:
                break

    if remaining:
        print("WARNING: The following IDs were not found in the val split:")
        for rid in remaining:
            print("   -", rid)

    for sid in sample_ids:
        if sid not in found:
            continue
        img_cpu, msk_cpu, logit_cpu = found[sid]
        visualize_triplet(
            img=img_cpu,
            msk=msk_cpu,
            logit=logit_cpu,
            sample_id=sid,
            outdir=args.outdir,
            threshold=args.threshold,
            suffix=f"ckpt{os.path.basename(args.ckpt)}",
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Visualize specific validation samples (image / GT / prediction)."
    )

    # Data / paths
    ap.add_argument("--root", type=str, default="../",
                    help="Project root containing metadata.csv and google/ folder.")
    ap.add_argument("--outdir", type=str, default="./val_viz",
                    help="Directory to save the PNG visualizations.")
    ap.add_argument("--model", type=str, choices=["unet", "swin-unet"], required=True,
                    help="Model architecture (matches train.py).")
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Path to checkpoint saved by utils.save_best.")

    ap.add_argument(
        "--ids", type=str, nargs="*",
        help="Optional list of sample IDs (filename stems). "
             "If omitted, uses the 3 default examples."
    )

    ap.add_argument("--img_size", type=int, default=400,
                    help="img_size argument used for SwinUNet backbone.")
    ap.add_argument("--resize_to", type=int, default=None,
                    help="Optional resize_to passed into PVSegDataset (default None).")

    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Threshold for binarizing predictions.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (must match train.py to reproduce val split).")

    args = ap.parse_args()
    main(args)
