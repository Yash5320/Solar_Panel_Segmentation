import os, cv2, random
import numpy as np
import torch
from torch.utils.data import Dataset

def _read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def _read_mask(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None: raise FileNotFoundError(path)
    m = (m > 127).astype(np.uint8)
    return m

class PVSegDataset(Dataset):
    def __init__(self, root, domain="google", df=None, split="train",
                 img_size=400, aug=True, normalize=None, resize_to=None):
        self.root = root
        self.domain = domain
        self.img_dir = os.path.join(root, domain, "img")
        self.msk_dir = os.path.join(root, domain, "mask")
        self.img_size = img_size
        self.aug = aug and (split == "train")
        self.normalize = normalize
        self.resize_to = resize_to 

        items = None
        if df is not None and "filename" in df.columns:
            items = df["filename"].tolist()

        if items is None:
            exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
            items = [f for f in os.listdir(self.img_dir) if f.lower().endswith(exts)]
            items.sort()

        kept = []
        for f in items:
            stem = os.path.splitext(f)[0]
            cand = [os.path.join(self.msk_dir, stem + e) for e in (".png", ".jpg", ".jpeg", ".tif", ".tiff")]
            if any(os.path.exists(p) for p in cand):
                kept.append(f)

        if len(kept) == 0:
            raise FileNotFoundError(f"No paired img/mask found under {self.img_dir} / {self.msk_dir}")

        self.items = kept

        for f in self.items[:3]:
            assert os.path.exists(os.path.join(self.img_dir, f)), f"Missing {f}"

    def __len__(self): return len(self.items)

    def _augment(self, img, msk):
        if random.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1, :]); msk = np.ascontiguousarray(msk[:, ::-1])
        if random.random() < 0.5:
            img = np.ascontiguousarray(img[::-1, :, :]); msk = np.ascontiguousarray(msk[::-1, :])
        if random.random() < 0.25:
            img = np.rot90(img).copy(); msk = np.rot90(msk).copy()
        if random.random() < 0.25:
            alpha = 0.85 + 0.3 * random.random()
            img = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        return img, msk

    def __getitem__(self, idx):
        fname = self.items[idx]
        ip = os.path.join(self.img_dir, fname)
        mp = os.path.join(self.msk_dir, os.path.splitext(fname)[0] + ".png")
        if not os.path.exists(mp):
            for ext in [".jpg", ".jpeg", ".tif"]:
                alt = os.path.join(self.msk_dir, os.path.splitext(fname)[0] + ext)
                if os.path.exists(alt): mp = alt; break

        img = _read_rgb(ip)
        msk = _read_mask(mp)

        if self.aug:
            img, msk = self._augment(img, msk)

        if self.resize_to is not None:
            img = cv2.resize(img, (self.resize_to, self.resize_to), interpolation=cv2.INTER_AREA)
            msk = cv2.resize(msk, (self.resize_to, self.resize_to), interpolation=cv2.INTER_NEAREST)
        
        img = img.astype(np.float32) / 255.0
        if self.normalize == "imagenet":
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
        
        img = torch.from_numpy(img.transpose(2,0,1)).float()
        msk = torch.from_numpy(msk[None, ...]).float()
        return img, msk, fname
