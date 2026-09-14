"""
[TV5] Xây dựng kiến trúc U-Net cho bài toán phân đoạn tổn thương da ISIC.

Kiến trúc:
    Encoder (down) -> Bottleneck -> Decoder (up) + skip connections
    Input : (B, 3, H, W)   - ảnh RGB
    Output: (B, 1, H, W)   - logits (chưa qua sigmoid)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2d -> BN -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling: MaxPool -> DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling: Upsample (hoặc ConvTranspose) -> Concat skip -> DoubleConv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                         kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Căn kích thước nếu lệch 1 pixel (do input không chia hết cho 16)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """1x1 conv để ra số kênh mong muốn."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net chuẩn (Ronneberger et al., 2015).
    Tham số:
        n_channels: số kênh ảnh vào (3 cho RGB)
        n_classes : số kênh output (1 cho binary segmentation)
        bilinear : True -> dùng Upsample bilinear, False -> dùng ConvTranspose2d
    """

       def __init__(
        self,
        n_channels=3,
        n_classes=1,
        bilinear=True,
        in_channels=None,
        out_channels=None
    ):
        super().__init__()

        if in_channels is not None:
            n_channels = in_channels

        if out_channels is not None:
            n_classes = out_channels

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Output
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    @torch.no_grad()
    def predict_mask(self, x, threshold=0.5):
        """Trả về mask nhị phân (0/1) - tiện cho inference."""
        self.eval()
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return (probs > threshold).float()


# ---------------------------------------------------------------------------
# Hàm tiện ích: khởi tạo model + đếm tham số
# ---------------------------------------------------------------------------
def build_unet(n_channels=3, n_classes=1, bilinear=True, device="cpu"):
    model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[UNet] Khởi tạo thành công | Tham số huấn luyện: {n_params:,}")
    return model


if __name__ == "__main__":
    # Smoke test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_unet(device=device)
    dummy = torch.randn(2, 3, 256, 256).to(device)
    out = model(dummy)
    print(f"Input : {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}")
