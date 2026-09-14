```python
"""
train_segmentation.py

Huấn luyện U-Net cho bài toán Skin Lesion Segmentation trên ISIC.

Input:
    data/images/train/
    data/masks/train/

Output:
    results/segmentation/
        best_model.pth
        last_model.pth
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF


# ============================================================
# CONFIG
# ============================================================

IMAGE_DIR = Path("data/images/train")
MASK_DIR = Path("data/masks/train")

OUTPUT_DIR = Path("results/segmentation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (256, 256)
NUM_EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATASET
# ============================================================

class ISICSegmentationDataset(Dataset):
    """
    Dataset ISIC cho U-Net.

    Mỗi ảnh có một segmentation mask tương ứng.
    """

    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.images = sorted(
            [
                file for file in image_dir.iterdir()
                if file.suffix.lower() in {
                    ".jpg",
                    ".jpeg",
                    ".png"
                }
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_path = self.images[index]

        # ----------------------------------------------------
        # Đọc ảnh
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        # ----------------------------------------------------
        # Tìm mask tương ứng
        # ----------------------------------------------------

        mask_path = self.mask_dir / f"{image_path.stem}.png"

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy mask: {mask_path}"
            )

        mask = Image.open(
            mask_path
        ).convert("L")

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.BILINEAR
        )

        mask = mask.resize(
            IMAGE_SIZE,
            Image.Resampling.NEAREST
        )

        # ----------------------------------------------------
        # Convert image → Tensor
        # ----------------------------------------------------

        image = TF.to_tensor(image)

        # ----------------------------------------------------
        # Convert mask → Tensor
        # ----------------------------------------------------

        mask = np.array(mask)

        # Pixel > 0 = lesion
        mask = (mask > 0).astype(np.float32)

        mask = torch.from_numpy(mask)

        # Thêm channel: H × W → 1 × H × W
        mask = mask.unsqueeze(0)

        return image, mask


# ============================================================
# U-NET
# ============================================================

class DoubleConv(nn.Module):
    """
    Hai lớp Convolution + BatchNorm + ReLU.
    """

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    U-Net đơn giản cho binary segmentation.
    """

    def __init__(self):

        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(
            256,
            512
        )

        # Decoder
        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            512,
            256
        )

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            256,
            128
        )

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            128,
            64
        )

        # Output
        self.output = nn.Conv2d(
            64,
            1,
            kernel_size=1
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        # ----------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------

        b = self.bottleneck(
            self.pool(e3)
        )

        # ----------------------------------------------------
        # Decoder + Skip Connections
        # ----------------------------------------------------

        d3 = self.up3(b)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        return self.output(d1)


# ============================================================
# DICE LOSS
# ============================================================

def dice_loss(prediction, target):

    prediction = torch.sigmoid(prediction)

    prediction = prediction.view(
        prediction.size(0),
        -1
    )

    target = target.view(
        target.size(0),
        -1
    )

    intersection = (
        prediction * target
    ).sum(dim=1)

    dice = (
        2 * intersection + 1e-6
    ) / (
        prediction.sum(dim=1)
        + target.sum(dim=1)
        + 1e-6
    )

    return 1 - dice.mean()


# ============================================================
# LOSS
# ============================================================

def combined_loss(prediction, target):

    bce = nn.functional.binary_cross_entropy_with_logits(
        prediction,
        target
    )

    dice = dice_loss(
        prediction,
        target
    )

    return bce + dice


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device
):

    model.train()

    total_loss = 0.0

    for images, masks in data_loader:

        images = images.to(device)
        masks = masks.to(device)

        # Forward
        predictions = model(images)

        # Loss
        loss = combined_loss(
            predictions,
            masks
        )

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("U-Net Segmentation Training")
    print("=" * 60)

    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = ISICSegmentationDataset(
        IMAGE_DIR,
        MASK_DIR
    )

    if len(dataset) == 0:
        raise RuntimeError(
            f"Không tìm thấy ảnh trong {IMAGE_DIR}"
        )

    print(
        f"Training images: {len(dataset)}"
    )

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = UNet()

    model.to(DEVICE)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_loss = float("inf")

    for epoch in range(NUM_EPOCHS):

        loss = train_one_epoch(
            model,
            data_loader,
            optimizer,
            DEVICE
        )

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"Loss: {loss:.4f}"
        )

        # Lưu model tốt nhất
        if loss < best_loss:

            best_loss = loss

            torch.save(
                model.state_dict(),
                OUTPUT_DIR / "best_model.pth"
            )

            print(
                "  → Saved best_model.pth"
            )

    # --------------------------------------------------------
    # Lưu model cuối
    # --------------------------------------------------------

    torch.save(
        model.state_dict(),
        OUTPUT_DIR / "last_model.pth"
    )

    print("\nTraining completed.")

    print(
        f"Best model: "
        f"{OUTPUT_DIR / 'best_model.pth'}"
    )

    print(
        f"Last model: "
        f"{OUTPUT_DIR / 'last_model.pth'}"
    )


if __name__ == "__main__":
    main()
```

### Cấu trúc thư mục

```text
project/
├── data/
│   ├── images/
│   │   └── train/
│   │       ├── ISIC_001.jpg
│   │       └── ...
│   │
│   └── masks/
│       └── train/
│           ├── ISIC_001.png
│           └── ...
│
├── train_segmentation.py
│
└── results/
    └── segmentation/
        ├── best_model.pth
        └── last_model.pth
```

### Cài thư viện

```bash
pip install torch torchvision pillow numpy
```

### Chạy

```bash
python train_segmentation.py
```

Model tốt nhất sẽ được lưu tại:

```text
results/segmentation/best_model.pth
```

**Lưu ý:** Bản này dùng `256×256`, `BCE + Dice Loss` và chưa có augmentation/validation. Đây là bản phù hợp để nhóm **chạy được pipeline trước**; sau khi chạy ổn, nên thêm validation và augmentation để có kết quả báo cáo đáng tin cậy hơn.
