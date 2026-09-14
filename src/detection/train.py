"""
train.py

Train Faster R-CNN cho bai toan phat hien ton thuong da.

Dataset:
    data/preprocessed/images/
        ISIC_XXXXXXX.jpg

    data/preprocessed/masks/
        ISIC_XXXXXXX_Segmentation.png

Bounding box duoc tao tu segmentation mask.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import functional as TF

from faster_rcnn import create_faster_rcnn


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# DATASET PATH
# ============================================================

# Dung dataset sau preprocessing
IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "preprocessed"
    / "images"
)

MASK_DIR = (
    PROJECT_ROOT
    / "data"
    / "preprocessed"
    / "masks"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "detection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL_PATH = (
    OUTPUT_DIR
    / "best_model.pth"
)

LAST_MODEL_PATH = (
    OUTPUT_DIR
    / "last_model.pth"
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# Faster R-CNN:
#   0 = background
#   1 = skin lesion
NUM_CLASSES = 2


NUM_EPOCHS = 10

BATCH_SIZE = 2

LEARNING_RATE = 0.005

MOMENTUM = 0.9

WEIGHT_DECAY = 0.0005


# ============================================================
# DATASET
# ============================================================

class ISICDetectionDataset(Dataset):
    """
    Dataset cho Faster R-CNN.

    Bounding box duoc tao tu segmentation mask.
    """

    def __init__(
        self,
        image_dir,
        mask_dir
    ):

        self.image_dir = Path(
            image_dir
        )

        self.mask_dir = Path(
            mask_dir
        )

        # ----------------------------------------------------
        # Tim tat ca anh
        # ----------------------------------------------------

        extensions = [
            "*.jpg",
            "*.jpeg",
            "*.JPG",
            "*.JPEG"
        ]

        self.image_files = []

        for extension in extensions:

            self.image_files.extend(
                self.image_dir.glob(
                    extension
                )
            )

        self.image_files = sorted(
            self.image_files
        )

        if len(self.image_files) == 0:

            raise RuntimeError(
                f"Khong tim thay anh trong: "
                f"{self.image_dir}"
            )

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(
            self.image_files
        )

    # ========================================================
    # FIND MASK
    # ========================================================

    def find_mask(
        self,
        image_path
    ):
        """
        Tim mask tuong ung voi anh.

        Vi du:

            ISIC_0000000.jpg

        Mask:

            ISIC_0000000_Segmentation.png
        """

        image_id = image_path.stem

        possible_masks = [

            self.mask_dir
            / f"{image_id}_Segmentation.png",

            self.mask_dir
            / f"{image_id}_segmentation.png",

            self.mask_dir
            / f"{image_id}.png"
        ]

        for mask_path in possible_masks:

            if mask_path.exists():

                return mask_path

        return None

    # ========================================================
    # MASK -> BOUNDING BOX
    # ========================================================

    def mask_to_bbox(
        self,
        mask
    ):
        """
        Chuyen segmentation mask
        thanh bounding box.

        Tra ve:

            [xmin, ymin, xmax, ymax]
        """

        ys, xs = np.where(
            mask > 0
        )

        # Khong co lesion
        if len(xs) == 0:

            return None

        xmin = xs.min()

        ymin = ys.min()

        xmax = xs.max()

        ymax = ys.max()

        return [
            float(xmin),
            float(ymin),
            float(xmax),
            float(ymax)
        ]

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(
        self,
        index
    ):

        # ----------------------------------------------------
        # Image path
        # ----------------------------------------------------

        image_path = (
            self.image_files[index]
        )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

        # ----------------------------------------------------
        # Find mask
        # ----------------------------------------------------

        mask_path = self.find_mask(
            image_path
        )

        if mask_path is None:

            raise FileNotFoundError(
                "Khong tim thay mask cho anh: "
                f"{image_path.name}"
            )

        # ----------------------------------------------------
        # Load mask
        # ----------------------------------------------------

        mask = np.array(
            Image.open(
                mask_path
            ).convert(
                "L"
            )
        )

        # ----------------------------------------------------
        # Convert mask to binary
        # ----------------------------------------------------

        mask = np.where(
            mask > 127,
            255,
            0
        ).astype(
            np.uint8
        )

        # ----------------------------------------------------
        # Create bounding box
        # ----------------------------------------------------

        bbox = self.mask_to_bbox(
            mask
        )

        # ----------------------------------------------------
        # Empty mask
        # ----------------------------------------------------

        if bbox is None:

            # Bounding box toi thieu
            bbox = [
                0.0,
                0.0,
                1.0,
                1.0
            ]

        # ----------------------------------------------------
        # Bounding boxes
        # ----------------------------------------------------

        boxes = torch.tensor(
            [bbox],
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # Labels
        #
        # 0 = background
        # 1 = skin lesion
        # ----------------------------------------------------

        labels = torch.tensor(
            [1],
            dtype=torch.int64
        )

        # ----------------------------------------------------
        # Bounding box area
        # ----------------------------------------------------

        area = torch.tensor(
            [
                (
                    bbox[2] - bbox[0]
                )
                *
                (
                    bbox[3] - bbox[1]
                )
            ],
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # Is crowd
        # ----------------------------------------------------

        iscrowd = torch.zeros(
            (1,),
            dtype=torch.int64
        )

        # ----------------------------------------------------
        # Image ID
        # ----------------------------------------------------

        image_id = torch.tensor(
            [index],
            dtype=torch.int64
        )

        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------

        target = {

            "boxes": boxes,

            "labels": labels,

            "image_id": image_id,

            "area": area,

            "iscrowd": iscrowd
        }

        # ----------------------------------------------------
        # Convert image to tensor
        # ----------------------------------------------------

        image = TF.to_tensor(
            image
        )

        return image, target


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_fn(batch):
    """
    Faster R-CNN can:
        list image
        list target
    """

    images, targets = zip(
        *batch
    )

    return (
        list(images),
        list(targets)
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
    epoch
):
    """
    Train model trong mot epoch.
    """

    model.train()

    total_loss = 0.0

    for batch_index, (
        images,
        targets
    ) in enumerate(
        data_loader
    ):

        # ----------------------------------------------------
        # Move images to device
        # ----------------------------------------------------

        images = [
            image.to(device)
            for image in images
        ]

        # ----------------------------------------------------
        # Move targets to device
        # ----------------------------------------------------

        targets = [

            {
                key: value.to(device)
                for key, value in target.items()
            }
            for target in targets
        ]

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        loss_dict = model(
            images,
            targets
        )

        # ----------------------------------------------------
        # Total loss
        # ----------------------------------------------------

        losses = sum(
            loss
            for loss in loss_dict.values()
        )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        optimizer.zero_grad()

        losses.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Save loss
        # ----------------------------------------------------

        total_loss += (
            losses.item()
        )

        # ----------------------------------------------------
        # Print progress
        # ----------------------------------------------------

        if (
            batch_index + 1
        ) % 10 == 0:

            print(
                f"Epoch {epoch} | "
                f"Batch {batch_index + 1}/"
                f"{len(data_loader)} | "
                f"Loss: "
                f"{losses.item():.4f}"
            )

    # --------------------------------------------------------
    # Average loss
    # --------------------------------------------------------

    average_loss = (
        total_loss
        /
        max(
            1,
            len(data_loader)
        )
    )

    return average_loss


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    path
):
    """
    Luu checkpoint cua model.
    """

    torch.save(
        {
            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "loss": loss
        },
        path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "FASTER R-CNN TRAINING"
    )

    print("=" * 60)

    print()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    print(
        "Device:",
        DEVICE
    )

    # --------------------------------------------------------
    # Dataset paths
    # --------------------------------------------------------

    print()

    print(
        "Image directory:",
        IMAGE_DIR
    )

    print(
        "Mask directory:",
        MASK_DIR
    )

    # --------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------

    print()

    print(
        "Epochs:",
        NUM_EPOCHS
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    # --------------------------------------------------------
    # Check image directory
    # --------------------------------------------------------

    if not IMAGE_DIR.exists():

        raise FileNotFoundError(
            "Image directory not found: "
            f"{IMAGE_DIR}"
        )

    # --------------------------------------------------------
    # Check mask directory
    # --------------------------------------------------------

    if not MASK_DIR.exists():

        raise FileNotFoundError(
            "Mask directory not found: "
            f"{MASK_DIR}"
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print()
    print(
        "Loading dataset..."
    )

    dataset = ISICDetectionDataset(
        IMAGE_DIR,
        MASK_DIR
    )

    print()

    print(
        "Number of images:",
        len(dataset)
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print()

    print(
        "Creating Faster R-CNN..."
    )

    model = create_faster_rcnn(
        num_classes=NUM_CLASSES,
        pretrained=True
    )

    model.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    params = [

        parameter

        for parameter
        in model.parameters()

        if parameter.requires_grad
    ]

    optimizer = torch.optim.SGD(
        params,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_loss = float(
        "inf"
    )

    history = []

    print()
    print(
        "Starting training..."
    )

    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        print()
        print(
            "=" * 60
        )

        print(
            f"Epoch {epoch}/{NUM_EPOCHS}"
        )

        print(
            "=" * 60
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        average_loss = train_one_epoch(
            model,
            data_loader,
            optimizer,
            DEVICE,
            epoch
        )

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        history.append(
            {
                "epoch": epoch,
                "loss": average_loss
            }
        )

        print()
        print(
            f"Epoch {epoch} "
            f"Average Loss: "
            f"{average_loss:.4f}"
        )

        # ----------------------------------------------------
        # Save last model
        # ----------------------------------------------------

        save_checkpoint(
            model,
            optimizer,
            epoch,
            average_loss,
            LAST_MODEL_PATH
        )

        print()

        print(
            "Last model saved."
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if average_loss < best_loss:

            best_loss = (
                average_loss
            )

            save_checkpoint(
                model,
                optimizer,
                epoch,
                average_loss,
                BEST_MODEL_PATH
            )

            print(
                "Best model saved."
            )

    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================

    history_file = (
        OUTPUT_DIR
        / "training_history.txt"
    )

    with open(
        history_file,
        "w",
        encoding="utf-8"
    ) as file:

        for item in history:

            file.write(
                f"Epoch {item['epoch']}: "
                f"Loss={item['loss']:.6f}\n"
            )

    # ========================================================
    # FINISH
    # ========================================================

    print()

    print("=" * 60)

    print(
        "TRAINING COMPLETED"
    )

    print("=" * 60)

    print()

    print(
        "Best model:",
        BEST_MODEL_PATH
    )

    print(
        "Last model:",
        LAST_MODEL_PATH
    )

    print(
        "History:",
        history_file
    )

    print()

    print(
        "Best loss:",
        f"{best_loss:.6f}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
