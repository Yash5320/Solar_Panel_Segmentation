import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.fuse = nn.Sequential(
            ConvBNReLU(out_ch + skip_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)

class SwinUNet(nn.Module):
    def __init__(self, backbone="swin_tiny_patch4_window7_224", out_ch=1, img_size=400):
        super().__init__()
        self.encoder = timm.create_model(
            backbone,
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=img_size,
        )
        self.enc_chs = self.encoder.feature_info.channels()

        self.proj1 = ConvBNReLU(self.enc_chs[0], 64)
        self.proj2 = ConvBNReLU(self.enc_chs[1], 128)
        self.proj3 = ConvBNReLU(self.enc_chs[2], 256)
        self.proj4 = ConvBNReLU(self.enc_chs[3], 512)

        self.center = ConvBNReLU(512, 512)
        self.up3 = UpBlock(512, 256, 256)
        self.up2 = UpBlock(256, 128, 128)
        self.up1 = UpBlock(128, 64, 64)

        self.head = nn.Conv2d(64, out_ch, 1)

    @torch.no_grad()
    def _to_nchw(self, t: torch.Tensor, c_expected: int) -> torch.Tensor:
        if t.ndim == 4 and t.shape[1] != c_expected and t.shape[-1] == c_expected:
            t = t.permute(0, 3, 1, 2).contiguous()
        return t

    def forward(self, x):
        f0, f1, f2, f3 = self.encoder(x)
        f0 = self._to_nchw(f0, self.enc_chs[0])
        f1 = self._to_nchw(f1, self.enc_chs[1])
        f2 = self._to_nchw(f2, self.enc_chs[2])
        f3 = self._to_nchw(f3, self.enc_chs[3])

        f0 = self.proj1(f0)
        f1 = self.proj2(f1)
        f2 = self.proj3(f2)
        f3 = self.proj4(f3)

        c = self.center(f3)

        d2 = self.up3(c, f2)
        d1 = self.up2(d2, f1)
        d0 = self.up1(d1, f0)

        logits = self.head(d0)
        
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits
