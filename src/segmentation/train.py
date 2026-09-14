"""
HUAN LUYEN U-NET CHO PHAN DOAN TON THUONG DA

Input:
    data/preprocessed/images/
    data/preprocessed/masks/

Output:
    results/segmentation/best_model.pth
    results/segmentation/last_model.pth
    results/segmentation/training_history.txt
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# ============================================================
# 1. CAU HINH DUONG DAN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "data" / "preprocessed" / "images"
MASK_DIR = PROJECT_ROOT / "data" / "preprocessed" / "masks"

RESULT_DIR = PROJECT_ROOT / "results" / "segmentation"

BEST_MODEL_PATH = RESULT_DIR / "best_model.pth"
LAST_MODEL_PATH = RESULT_DIR / "last_model.pth"
HISTORY_PATH = RESULT_DIR / "training_history.txt"


# ============================================================
# 2. CAU HINH HUAN LUYEN
# ============================================================

IMAGE_SIZE = 256

EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 0.001

TRAIN_RATIO = 0.8
VAL_RATIO = 0.2

RANDOM_SEED = 42


# ============================================================
# 3. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 4. DATASET
# ============================================================

class ISICSegmentationDataset(Dataset):
    """
    Dataset cho bai toan segmentation.

    Moi mau gom:
        image: anh ISIC
        mask : mat na vung ton thuong
    """

    def __init__(self, image_dir, mask_dir):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Khong tim thay thu muc anh: {self.image_dir}"
            )

        if not self.mask_dir.exists():
            raise FileNotFoundError(
                f"Khong tim thay thu muc mask: {self.mask_dir}"
            )

        # Lay tat ca file anh va tranh trung lap
        image_files = []

        for path in self.image_dir.iterdir():
            if (
                path.is_file()
                and path.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ):
                image_files.append(path)

        self.image_files = sorted(
            image_files,
            key=lambda x: x.name.lower()
        )

        if len(self.image_files) == 0:
            raise RuntimeError(
                f"Khong co anh trong: {self.image_dir}"
            )

        # Kiem tra mask tuong ung
        valid_pairs = []

        for image_path in self.image_files:

            mask_path = self.find_mask(image_path)

            if mask_path is not None:
                valid_pairs.append(
                    (image_path, mask_path)
                )

        self.samples = valid_pairs

        if len(self.samples) == 0:
            raise RuntimeError(
                "Khong tim thay cap image-mask nao."
            )

    # --------------------------------------------------------
    # Tim mask tuong ung
    # --------------------------------------------------------

    def find_mask(self, image_path):

        stem = image_path.stem

        possible_names = [
            f"{stem}_Segmentation.png",
            f"{stem}_segmentation.png",
            f"{stem}.png",
        ]

        for name in possible_names:

            mask_path = self.mask_dir / name

            if mask_path.exists():
                return mask_path

        return None

    # --------------------------------------------------------
    # So luong mau
    # --------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    # --------------------------------------------------------
    # Doc 1 mau
    # --------------------------------------------------------

    def __getitem__(self, index):

        image_path, mask_path = self.samples[index]

        # Doc anh
        image = Image.open(image_path).convert("RGB")

        # Doc mask
        mask = Image.open(mask_path).convert("L")

        # Resize
        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR
        )

        mask = mask.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.NEAREST
        )

        # Chuyen numpy
        image = np.array(image, dtype=np.float32)
        mask = np.array(mask, dtype=np.float32)

        # Chuan hoa image ve [0, 1]
        image = image / 255.0

        # Chuyen mask thanh 0 va 1
        mask = mask / 255.0
        mask = (mask > 0.5).astype(np.float32)

        # HWC -> CHW
        image = np.transpose(
            image,
            (2, 0, 1)
        )

        # Mask them channel
        mask = np.expand_dims(
            mask,
            axis=0
        )

        # Tensor
        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        mask = torch.tensor(
            mask,
            dtype=torch.float32
        )

        return image, mask


# ============================================================
# 5. DICE LOSS
# ============================================================

class DiceLoss(nn.Module):
    """
    Dice Loss dung de do muc do trung khop giua
    mask du doan va mask thuc te.
    """

    def __init__(self, smooth=1.0):
        super().__init__()

        self.smooth = smooth

    def forward(self, predictions, targets):

        predictions = torch.sigmoid(predictions)

        predictions = predictions.contiguous().view(
            predictions.size(0),
            -1
        )

        targets = targets.contiguous().view(
            targets.size(0),
            -1
        )

        intersection = (
            predictions * targets
        ).sum(dim=1)

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            predictions.sum(dim=1)
            + targets.sum(dim=1)
            + self.smooth
        )

        return 1.0 - dice.mean()


# ============================================================
# 6. BCE + DICE LOSS
# ============================================================

class BCEDiceLoss(nn.Module):
    """
    Ket hop BCEWithLogitsLoss va DiceLoss.
    """

    def __init__(self):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, predictions, targets):

        bce_loss = self.bce(
            predictions,
            targets
        )

        dice_loss = self.dice(
            predictions,
            targets
        )

        return bce_loss + dice_loss


# ============================================================
# 7. U-NET
# ============================================================

class DoubleConv(nn.Module):
    """
    Hai lop convolution lien tiep.
    """

    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):

        return self.block(x)


class UNet(nn.Module):
    """
    Mo hinh U-Net cho segmentation.
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=1
    ):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(
            in_channels,
            64
        )

        self.enc2 = DoubleConv(
            64,
            128
        )

        self.enc3 = DoubleConv(
            128,
            256
        )

        self.enc4 = DoubleConv(
            256,
            512
        )

        # Bottleneck
        self.bottleneck = DoubleConv(
            512,
            1024
        )

        # Pooling
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(
            1024,
            512,
            kernel_size=2,
            stride=2
        )

        self.dec4 = DoubleConv(
            1024,
            512
        )

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
        self.out = nn.Conv2d(
            64,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        # Bottleneck
        b = self.bottleneck(
            self.pool(e4)
        )

        # Decoder
        d4 = self.up4(b)

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(d4)

        d3 = self.up3(d4)

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

        output = self.out(d1)

        return output


# ============================================================
# 8. TRAIN 1 EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    epoch
):

    model.train()

    running_loss = 0.0

    total_batches = len(loader)

    for batch_index, (images, masks) in enumerate(
        loader,
        start=1
    ):

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # Xoa gradient cu
        optimizer.zero_grad()

        # Forward
        predictions = model(images)

        # Loss
        loss = criterion(
            predictions,
            masks
        )

        # Backward
        loss.backward()

        # Update weight
        optimizer.step()

        running_loss += loss.item()

        # Hien thi tien do
        if (
            batch_index % 5 == 0
            or batch_index == total_batches
        ):

            print(
                f"Epoch {epoch} | "
                f"Batch {batch_index}/{total_batches} | "
                f"Loss: {loss.item():.4f}"
            )

    average_loss = (
        running_loss / total_batches
    )

    return average_loss


# ============================================================
# 9. VALIDATE
# ============================================================

def validate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            predictions = model(images)

            loss = criterion(
                predictions,
                masks
            )

            running_loss += loss.item()

    average_loss = (
        running_loss / len(loader)
    )

    return average_loss


# ============================================================
# 10. MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("U-NET TRAINING")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Image directory: {IMAGE_DIR}"
    )

    print(
        f"Mask directory: {MASK_DIR}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print()

    # --------------------------------------------------------
    # Tao thu muc ket qua
    # --------------------------------------------------------

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("Loading dataset...")

    dataset = ISICSegmentationDataset(
        IMAGE_DIR,
        MASK_DIR
    )

    print(
        f"Number of valid image-mask pairs: "
        f"{len(dataset)}"
    )

    if len(dataset) < 2:

        print(
            "[ERROR] Can it nhat 2 cap image-mask."
        )

        return 1

    # --------------------------------------------------------
    # Chia train / validation
    # --------------------------------------------------------

    torch.manual_seed(
        RANDOM_SEED
    )

    train_size = int(
        TRAIN_RATIO * len(dataset)
    )

    val_size = (
        len(dataset) - train_size
    )

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(
            RANDOM_SEED
        )
    )

    print(
        f"Training samples: {len(train_dataset)}"
    )

    print(
        f"Validation samples: {len(val_dataset)}"
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print()
    print("Creating U-Net...")

    model = UNet(
        in_channels=3,
        out_channels=1
    )

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = BCEDiceLoss()

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

    print()
    print("Starting training...")
    print()

    best_val_loss = float("inf")

    history = []

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            epoch
        )

        val_loss = validate(
            model,
            val_loader,
            criterion
        )

        print(
            f"Epoch {epoch} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        print()

        history.append(
            f"Epoch {epoch}: "
            f"Train Loss={train_loss:.6f}, "
            f"Val Loss={val_loss:.6f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                BEST_MODEL_PATH
            )

            print(
                f"[OK] Saved best model: "
                f"{BEST_MODEL_PATH}"
            )

            print()

    # --------------------------------------------------------
    # Save last model
    # --------------------------------------------------------

    torch.save(
        model.state_dict(),
        LAST_MODEL_PATH
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    with open(
        HISTORY_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "U-NET TRAINING HISTORY\n"
        )

        file.write(
            "=" * 50
            + "\n"
        )

        file.write(
            f"Device: {DEVICE}\n"
        )

        file.write(
            f"Dataset size: {len(dataset)}\n"
        )

        file.write(
            f"Train size: {len(train_dataset)}\n"
        )

        file.write(
            f"Validation size: {len(val_dataset)}\n"
        )

        file.write(
            f"Epochs: {EPOCHS}\n"
        )

        file.write(
            f"Batch size: {BATCH_SIZE}\n"
        )

        file.write(
            "\n"
        )

        for line in history:

            file.write(
                line + "\n"
            )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print("=" * 70)
    print("U-NET TRAINING HOAN TAT")
    print("=" * 70)

    print(
        f"Best model : {BEST_MODEL_PATH}"
    )

    print(
        f"Last model : {LAST_MODEL_PATH}"
    )

    print(
        f"History    : {HISTORY_PATH}"
    )

    return 0


# ============================================================
# 11. RUN
# ============================================================

if __name__ == "__main__":

    try:

        exit_code = main()

        sys.exit(exit_code)

    except KeyboardInterrupt:

        print()
        print(
            "[WARNING] Da dung chuong trinh."
        )

        sys.exit(1)

    except Exception as error:

        print()
        print(
            f"[ERROR] {type(error).__name__}: {error}"
        )

        sys.exit(1)
