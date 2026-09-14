<<<<<<< HEAD
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

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
from pathlib import Path

import numpy as np
import torch
from PIL import Image
<<<<<<< HEAD

from torch.utils.data import Dataset, DataLoader

=======
from torch.utils.data import Dataset, DataLoader
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
from torchvision.transforms import functional as TF

from faster_rcnn import create_faster_rcnn


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

<<<<<<< HEAD

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
=======
IMAGE_DIR = PROJECT_ROOT / "data" / "images" / "train"
MASK_DIR = PROJECT_ROOT / "data" / "masks" / "train"
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "detection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

<<<<<<< HEAD

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
BEST_MODEL_PATH = (
    OUTPUT_DIR
    / "best_model.pth"
)

<<<<<<< HEAD

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
LAST_MODEL_PATH = (
    OUTPUT_DIR
    / "last_model.pth"
)

<<<<<<< HEAD

# ============================================================
# TRAINING CONFIGURATION
# ============================================================

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

<<<<<<< HEAD

# Faster R-CNN:
#   0 = background
#   1 = skin lesion
NUM_CLASSES = 2


=======
NUM_CLASSES = 2

>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
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

<<<<<<< HEAD
        # ----------------------------------------------------
        # Tim tat ca anh
        # ----------------------------------------------------

        extensions = [
            "*.jpg",
            "*.jpeg",
            "*.JPG",
            "*.JPEG"
=======
        extensions = [
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.JPG",
            "*.JPEG",
            "*.PNG"
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
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
<<<<<<< HEAD
                f"Khong tim thay anh trong: "
                f"{self.image_dir}"
            )

    # ========================================================
    # LENGTH
    # ========================================================

=======
                f"No images found in {self.image_dir}"
            )

>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    def __len__(self):

        return len(
            self.image_files
        )

<<<<<<< HEAD
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

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    def mask_to_bbox(
        self,
        mask
    ):
        """
<<<<<<< HEAD
        Chuyen segmentation mask
        thanh bounding box.

        Tra ve:

            [xmin, ymin, xmax, ymax]
=======
        Chuyen mask thanh bounding box.
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
        """

        ys, xs = np.where(
            mask > 0
        )

<<<<<<< HEAD
        # Khong co lesion
=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
        if len(xs) == 0:

            return None

        xmin = xs.min()
<<<<<<< HEAD

        ymin = ys.min()

        xmax = xs.max()

=======
        ymin = ys.min()

        xmax = xs.max()
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
        ymax = ys.max()

        return [
            float(xmin),
            float(ymin),
            float(xmax),
            float(ymax)
        ]

<<<<<<< HEAD
    # ========================================================
    # GET ITEM
    # ========================================================

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    def __getitem__(
        self,
        index
    ):

<<<<<<< HEAD
        # ----------------------------------------------------
        # Image path
        # ----------------------------------------------------

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
        image_path = (
            self.image_files[index]
        )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(
            image_path
<<<<<<< HEAD
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
=======
        ).convert("RGB")
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1

        # ----------------------------------------------------
        # Load mask
        # ----------------------------------------------------

<<<<<<< HEAD
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
=======
        mask_path = (
            self.mask_dir
            / f"{image_path.stem}.png"
        )

        if not mask_path.exists():

            raise FileNotFoundError(
                f"Mask not found: {mask_path}"
            )

        mask = np.array(
            Image.open(
                mask_path
            ).convert("L")
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
        )

        # ----------------------------------------------------
        # Create bounding box
        # ----------------------------------------------------

        bbox = self.mask_to_bbox(
            mask
        )

<<<<<<< HEAD
        # ----------------------------------------------------
        # Empty mask
        # ----------------------------------------------------

        if bbox is None:

            # Bounding box toi thieu
=======
        if bbox is None:

            # Bounding box gia truong hop
            # mask rong
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
            bbox = [
                0.0,
                0.0,
                1.0,
                1.0
            ]

<<<<<<< HEAD
        # ----------------------------------------------------
        # Bounding boxes
        # ----------------------------------------------------

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
        boxes = torch.tensor(
            [bbox],
            dtype=torch.float32
        )

<<<<<<< HEAD
        # ----------------------------------------------------
        # Labels
        #
        # 0 = background
        # 1 = skin lesion
        # ----------------------------------------------------

=======
        # Class 1 = skin lesion
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
        labels = torch.tensor(
            [1],
            dtype=torch.int64
        )

        # ----------------------------------------------------
<<<<<<< HEAD
        # Bounding box area
=======
        # Area
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
        # ----------------------------------------------------

        area = torch.tensor(
            [
<<<<<<< HEAD
                (
                    bbox[2] - bbox[0]
                )
                *
                (
                    bbox[3] - bbox[1]
                )
=======
                (bbox[2] - bbox[0])
                *
                (bbox[3] - bbox[1])
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
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
<<<<<<< HEAD

            "boxes": boxes,

            "labels": labels,

            "image_id": image_id,

            "area": area,

=======
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
            "iscrowd": iscrowd
        }

        # ----------------------------------------------------
<<<<<<< HEAD
        # Convert image to tensor
=======
        # Tensor image
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
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
<<<<<<< HEAD
    Faster R-CNN can:
        list image
        list target
=======
    Faster R-CNN can nhan list image
    va list target.
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
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
<<<<<<< HEAD
        # Move images to device
=======
        # Move data to device
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
        # ----------------------------------------------------

        images = [
            image.to(device)
            for image in images
        ]

<<<<<<< HEAD
        # ----------------------------------------------------
        # Move targets to device
        # ----------------------------------------------------

        targets = [

=======
        targets = [
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
            {
                key: value.to(device)
                for key, value in target.items()
            }
<<<<<<< HEAD

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
            for target in targets
        ]

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        loss_dict = model(
            images,
            targets
        )

<<<<<<< HEAD
        # ----------------------------------------------------
        # Total loss
        # ----------------------------------------------------

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
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

<<<<<<< HEAD
        # ----------------------------------------------------
        # Print progress
        # ----------------------------------------------------

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
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

<<<<<<< HEAD
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
=======
    average_loss = (
        total_loss
        /
        max(1, len(data_loader))
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
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
<<<<<<< HEAD
    Luu checkpoint cua model.
=======
    Luu checkpoint.
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    """

    torch.save(
        {
            "epoch": epoch,
<<<<<<< HEAD

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

=======
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
            "loss": loss
        },
        path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
<<<<<<< HEAD

    print(
        "FASTER R-CNN TRAINING"
    )

=======
    print(
        "FASTER R-CNN TRAINING"
    )
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    print("=" * 60)

    print()

<<<<<<< HEAD
    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    print(
        "Device:",
        DEVICE
    )

<<<<<<< HEAD
    # --------------------------------------------------------
    # Dataset paths
    # --------------------------------------------------------

    print()

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    print(
        "Image directory:",
        IMAGE_DIR
    )

    print(
        "Mask directory:",
        MASK_DIR
    )

<<<<<<< HEAD
    # --------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------

    print()

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    print(
        "Epochs:",
        NUM_EPOCHS
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    # --------------------------------------------------------
<<<<<<< HEAD
    # Check image directory
=======
    # Check directories
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    # --------------------------------------------------------

    if not IMAGE_DIR.exists():

        raise FileNotFoundError(
<<<<<<< HEAD
            "Image directory not found: "
            f"{IMAGE_DIR}"
        )

    # --------------------------------------------------------
    # Check mask directory
    # --------------------------------------------------------

    if not MASK_DIR.exists():

        raise FileNotFoundError(
            "Mask directory not found: "
=======
            f"Image directory not found: "
            f"{IMAGE_DIR}"
        )

    if not MASK_DIR.exists():

        raise FileNotFoundError(
            f"Mask directory not found: "
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
            f"{MASK_DIR}"
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print()
<<<<<<< HEAD

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    print(
        "Loading dataset..."
    )

    dataset = ISICDetectionDataset(
        IMAGE_DIR,
        MASK_DIR
    )

<<<<<<< HEAD
    print()

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
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
<<<<<<< HEAD
    # Create model
    # --------------------------------------------------------

    print()

=======
    # Model
    # --------------------------------------------------------

    print()
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
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
<<<<<<< HEAD

        parameter

        for parameter
        in model.parameters()

=======
        parameter
        for parameter in model.parameters()
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
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

<<<<<<< HEAD
    best_loss = float(
        "inf"
    )
=======
    best_loss = float("inf")
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1

    history = []

    print()
<<<<<<< HEAD

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    print(
        "Starting training..."
    )

<<<<<<< HEAD
    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        print()
<<<<<<< HEAD

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
        print(
            "=" * 60
        )

        print(
            f"Epoch {epoch}/{NUM_EPOCHS}"
        )

        print(
            "=" * 60
        )

<<<<<<< HEAD
        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
        average_loss = train_one_epoch(
            model,
            data_loader,
            optimizer,
            DEVICE,
            epoch
        )

<<<<<<< HEAD
        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
        history.append(
            {
                "epoch": epoch,
                "loss": average_loss
            }
        )

        print()
<<<<<<< HEAD

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
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

<<<<<<< HEAD
        print()

        print(
            "Last model saved."
        )

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if average_loss < best_loss:

<<<<<<< HEAD
            best_loss = (
                average_loss
            )
=======
            best_loss = average_loss
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1

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

<<<<<<< HEAD
    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================
=======
    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1

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

<<<<<<< HEAD
    # ========================================================
    # FINISH
    # ========================================================

    print()

    print("=" * 60)

    print(
        "TRAINING COMPLETED"
    )

=======
    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        "TRAINING COMPLETED"
    )
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
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

<<<<<<< HEAD
    print()

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    print(
        "Best loss:",
        f"{best_loss:.6f}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
<<<<<<< HEAD

    main()
=======
    main()
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
