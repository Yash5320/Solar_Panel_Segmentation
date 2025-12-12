import os, time, argparse, pandas as pd, torch
from torch.utils.data import DataLoader
from dataset import PVSegDataset
from metrics import seg_metrics
from models.unet import UNet
from models.swin_unet import SwinUNet
from tqdm import tqdm

def get_model(model_name, ckpt):
    if model_name=="unet": m=UNet(in_ch=3,out_ch=1)
    else: m=SwinUNet(backbone="swin_tiny_patch4_window7_224", out_ch=1)
    sd = torch.load(ckpt, map_location="cpu")["state_dict"]
    m.load_state_dict(sd, strict=True)
    return m

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = pd.read_csv(os.path.join(args.root, "metadata.csv"))
    ign_df = meta.copy()
    ds = PVSegDataset(args.root, domain="google", df=ign_df, split="val", aug=False, normalize=("imagenet" if args.model=="swin-unet" else None), resize_to=256)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    model = get_model(args.model, args.ckpt).to(device).eval()

    latencies = []
    agg = {"precision":0,"recall":0,"f1":0,"iou":0}; n=0
    with torch.no_grad():
        for imgs, msks, fnames in tqdm(dl, desc="Testing IGN"):
            imgs, msks = imgs.to(device), msks.to(device)

            t0 = time.perf_counter()
            logits = model(imgs)
            torch.cuda.synchronize() if device.type=="cuda" else None
            dt = (time.perf_counter() - t0)
            latencies.append(dt)

            m = seg_metrics(logits, msks)
            for k in agg: agg[k] += m[k]
            n += 1

    for k in agg: agg[k] /= max(1,n)
    ms = 1000.0 * (sum(latencies)/len(latencies))
    print(f"[IGN] IoU={agg['iou']:.4f} F1={agg['f1']:.4f} P={agg['precision']:.4f} R={agg['recall']:.4f}")
    print(f"[IGN] Inference time: {ms:.2f} ms/img")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="../")
    ap.add_argument("--model", type=str, choices=["unet","swin-unet"], required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    args = ap.parse_args()
    main(args)
