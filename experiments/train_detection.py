```python
"""
train_detection.py

Huấn luyện Faster R-CNN để phát hiện skin lesion trên ảnh ISIC.

Input:
    data/images/train/
    data/masks/train/

Mask được dùng để tạo Bounding Box Ground Truth.

Output:
    results/detection/
        best_model.pth
        last_model.pth
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF


# ============================================================
# CONFIG
# ============================================================

IMAGE_DIR = Path("data/images/train")
MASK_DIR = Path("data/masks/train")

OUTPUT_DIR = Path("results/detection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 2       # Background + Lesion
NUM_EPOCHS = 10
BATCH_SIZE = 2
LEARNING_RATE = 0.005

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATASET
# ============================================================

class ISICDetectionDataset(Dataset):
    """
    Dataset ISIC cho Faster R-CNN.

    Bounding Box được tạo tự động từ segmentation mask.
    """

    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.images = sorted(
            [
                file for file in image_dir.iterdir()
                if file.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_path = self.images[index]

        # ----------------------------------------------------
        # Đọc ảnh
        # ----------------------------------------------------

        image = Image.open(image_path).convert("RGB")

        # ----------------------------------------------------
        # Tìm mask tương ứng
        # ----------------------------------------------------

        mask_path = self.mask_dir / f"{image_path.stem}.png"

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy mask: {mask_path}"
            )

        mask = np.array(
            Image.open(mask_path).convert("L")
        )

        # ----------------------------------------------------
        # Tạo Bounding Box từ mask
        # ----------------------------------------------------

        ys, xs = np.where(mask > 0)

        if len(xs) == 0:
            # Không có lesion
            boxes = torch.zeros(
                (0, 4),
                dtype=torch.float32
            )

            labels = torch.zeros(
                (0,),
                dtype=torch.int64
            )

        else:
            xmin = xs.min()
            xmax = xs.max()
            ymin = ys.min()
            ymax = ys.max()

            boxes = torch.tensor(
                [[xmin, ymin, xmax, ymax]],
                dtype=torch.float32
            )

            # Class 1 = Lesion
            labels = torch.tensor(
                [1],
                dtype=torch.int64
            )

        # ----------------------------------------------------
        # Convert Image → Tensor
        # ----------------------------------------------------

        image = TF.to_tensor(image)

        # ----------------------------------------------------
        # Target cho Faster R-CNN
        # ----------------------------------------------------

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index]),
        }

        return image, target


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_fn(batch):
    """
    Faster R-CNN cần xử lý ảnh có kích thước khác nhau.
    """

    return tuple(zip(*batch))


# ============================================================
# MODEL
# ============================================================

def create_model():

    # Model pretrained trên COCO
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

    model = fasterrcnn_resnet50_fpn(
        weights=weights
    )

    # Thay Detection Head
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        NUM_CLASSES
    )

    return model


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
):

    model.train()

    total_loss = 0.0

    for images, targets in data_loader:

        images = [
            image.to(device)
            for image in images
        ]

        targets = [
            {
                key: value.to(device)
                for key, value in target.items()
            }
            for target in targets
        ]

        # Tính loss
        loss_dict = model(
            images,
            targets
        )

        losses = sum(
            loss for loss in loss_dict.values()
        )

        # Backpropagation
        optimizer.zero_grad()

        losses.backward()

        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(data_loader)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("Faster R-CNN Training")
    print("=" * 60)

    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = ISICDetectionDataset(
        IMAGE_DIR,
        MASK_DIR
    )

    if len(dataset) == 0:
        raise RuntimeError(
            f"Không tìm thấy ảnh trong {IMAGE_DIR}"
        )

    print(f"Training images: {len(dataset)}")

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model()

    model.to(DEVICE)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.SGD(
        params,
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=0.0005,
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
            DEVICE,
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

            print("  → Saved best_model.pth")

    # --------------------------------------------------------
    # Lưu model cuối cùng
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

### Cấu trúc dữ liệu cần có

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
├── train_detection.py
│
└── results/
    └── detection/
```

### Cài thư viện

```bash
pip install torch torchvision pillow numpy
```

### Chạy

```bash
python train_detection.py
```

Model sau khi train:

```text
results/
└── detection/
    ├── best_model.pth
    └── last_model.pth
```

**Lưu ý quan trọng:** Script này tạo Bounding Box trực tiếp từ **Ground Truth Mask**. Vì vậy dataset train phải có ảnh và mask cùng ID, ví dụ `ISIC_001.jpg` ↔ `ISIC_001.png`.
