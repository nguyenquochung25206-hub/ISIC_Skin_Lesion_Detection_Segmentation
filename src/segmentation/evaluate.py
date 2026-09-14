"""
U-NET SEGMENTATION EVALUATION

Danh gia mo hinh U-Net.

Input:
    data/preprocessed/images/
    data/preprocessed/masks/

Model:
    results/segmentation/best_model.pth

Output:
    results/segmentation/evaluation/
        segmentation_results.json
        segmentation_metrics.json

Metrics:
    - Dice Score
    - IoU
    - Precision
    - Recall
"""


from pathlib import Path
import json
import sys

import numpy as np

import torch
import torch.nn as nn

from PIL import Image


# ============================================================
# 1. IMPORT UNET
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


# UNet duoc dinh nghia trong train.py
from train import UNet


# ============================================================
# 2. CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Dataset da preprocessing
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


# Model
MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "segmentation"
    / "best_model.pth"
)


# Output
OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "segmentation"
    / "evaluation"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "segmentation_results.json"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "segmentation_metrics.json"
)


# ============================================================
# 3. MODEL CONFIG
# ============================================================

IMAGE_SIZE = 256

THRESHOLD = 0.5

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 4. FIND MASK
# ============================================================

def find_mask(image_path):
    """
    Tim mask tuong ung voi anh.

    Vi du:

        ISIC_0000000.jpg

    se tim:

        ISIC_0000000_Segmentation.png
        ISIC_0000000_segmentation.png
        ISIC_0000000.png
    """

    image_stem = image_path.stem

    candidates = [

        MASK_DIR
        / f"{image_stem}_Segmentation.png",

        MASK_DIR
        / f"{image_stem}_segmentation.png",

        MASK_DIR
        / f"{image_stem}.png"
    ]

    for mask_path in candidates:

        if mask_path.exists():

            return mask_path

    return None


# ============================================================
# 5. CREATE MODEL
# ============================================================

def create_model():

    model = UNet(
        in_channels=3,
        out_channels=1
    )

    return model


# ============================================================
# 6. LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Khong tim thay model: {MODEL_PATH}"
        )

    print(
        "Loading U-Net model..."
    )

    model = create_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Lay state dict
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

        elif "state_dict" in checkpoint:

            state_dict = (
                checkpoint[
                    "state_dict"
                ]
            )

        elif "model_state" in checkpoint:

            state_dict = (
                checkpoint[
                    "model_state"
                ]
            )

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    model.load_state_dict(
        state_dict
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        "Model loaded successfully."
    )

    return model


# ============================================================
# 7. LOAD IMAGE
# ============================================================

def load_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        Image.Resampling.BILINEAR
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Normalize [0, 1]
    image_array = (
        image_array / 255.0
    )

    # HWC -> CHW
    image_array = np.transpose(
        image_array,
        (2, 0, 1)
    )

    tensor = torch.tensor(
        image_array,
        dtype=torch.float32
    )

    # Them batch dimension
    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.to(
        DEVICE
    )

    return tensor


# ============================================================
# 8. LOAD GROUND TRUTH MASK
# ============================================================

def load_mask(mask_path):

    mask = Image.open(
        mask_path
    ).convert("L")

    mask = mask.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        Image.Resampling.NEAREST
    )

    mask_array = np.array(
        mask,
        dtype=np.uint8
    )

    # Chuyen thanh binary
    mask_binary = (
        mask_array > 127
    ).astype(
        np.uint8
    )

    return mask_binary


# ============================================================
# 9. PREDICT MASK
# ============================================================

@torch.no_grad()
def predict_mask(
    model,
    image_tensor
):

    output = model(
        image_tensor
    )

    # Sigmoid
    probability = torch.sigmoid(
        output
    )

    probability = (
        probability
        .cpu()
        .numpy()[0, 0]
    )

    # Threshold
    predicted_mask = (
        probability >= THRESHOLD
    ).astype(
        np.uint8
    )

    return predicted_mask


# ============================================================
# 10. CALCULATE DICE
# ============================================================

def calculate_dice(
    prediction,
    ground_truth
):

    prediction = (
        prediction > 0
    )

    ground_truth = (
        ground_truth > 0
    )

    intersection = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    prediction_area = prediction.sum()

    ground_truth_area = ground_truth.sum()

    denominator = (
        prediction_area
        + ground_truth_area
    )

    if denominator == 0:

        return 1.0

    dice = (
        2.0
        * intersection
        / denominator
    )

    return float(
        dice
    )


# ============================================================
# 11. CALCULATE IOU
# ============================================================

def calculate_iou(
    prediction,
    ground_truth
):

    prediction = (
        prediction > 0
    )

    ground_truth = (
        ground_truth > 0
    )

    intersection = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    union = np.logical_or(
        prediction,
        ground_truth
    ).sum()

    if union == 0:

        return 1.0

    iou = (
        intersection
        / union
    )

    return float(
        iou
    )


# ============================================================
# 12. CALCULATE PIXEL METRICS
# ============================================================

def calculate_pixel_metrics(
    prediction,
    ground_truth
):

    prediction = (
        prediction > 0
    )

    ground_truth = (
        ground_truth > 0
    )

    # --------------------------------------------------------
    # TP
    # --------------------------------------------------------

    true_positive = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    # --------------------------------------------------------
    # FP
    # --------------------------------------------------------

    false_positive = np.logical_and(
        prediction,
        np.logical_not(
            ground_truth
        )
    ).sum()

    # --------------------------------------------------------
    # FN
    # --------------------------------------------------------

    false_negative = np.logical_and(
        np.logical_not(
            prediction
        ),
        ground_truth
    ).sum()

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = (
        true_positive
        /
        max(
            1,
            true_positive
            + false_positive
        )
    )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall = (
        true_positive
        /
        max(
            1,
            true_positive
            + false_negative
        )
    )

    return (
        int(true_positive),
        int(false_positive),
        int(false_negative),
        float(precision),
        float(recall)
    )


# ============================================================
# 13. EVALUATE ONE IMAGE
# ============================================================

def evaluate_image(
    model,
    image_path
):

    # --------------------------------------------------------
    # Find mask
    # --------------------------------------------------------

    mask_path = find_mask(
        image_path
    )

    if mask_path is None:

        return {
            "image": image_path.name,
            "status": "missing_mask"
        }

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image_tensor = load_image(
        image_path
    )

    # --------------------------------------------------------
    # Load ground truth
    # --------------------------------------------------------

    ground_truth = load_mask(
        mask_path
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = predict_mask(
        model,
        image_tensor
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    dice = calculate_dice(
        prediction,
        ground_truth
    )

    iou = calculate_iou(
        prediction,
        ground_truth
    )

    (
        true_positive,
        false_positive,
        false_negative,
        precision,
        recall
    ) = calculate_pixel_metrics(
        prediction,
        ground_truth
    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {

        "image": image_path.name,

        "mask": mask_path.name,

        "status": "evaluated",

        "dice": dice,

        "iou": iou,

        "precision": precision,

        "recall": recall,

        "true_positive": true_positive,

        "false_positive": false_positive,

        "false_negative": false_negative
    }


# ============================================================
# 14. GET IMAGE FILES
# ============================================================

def get_image_files():

    if not IMAGE_DIR.exists():

        return []

    image_files = []

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png"
    }

    # Dung iterdir de tranh trung file
    for path in IMAGE_DIR.iterdir():

        if not path.is_file():

            continue

        if path.suffix.lower() in valid_extensions:

            image_files.append(
                path
            )

    return sorted(
        image_files,
        key=lambda x: x.name.lower()
    )


# ============================================================
# 15. CALCULATE SUMMARY
# ============================================================

def calculate_summary(
    results
):

    valid_results = [

        result

        for result in results

        if result.get("status")
        == "evaluated"

    ]

    if len(valid_results) == 0:

        return {

            "number_of_images": 0,

            "mean_dice": 0.0,

            "mean_iou": 0.0,

            "mean_precision": 0.0,

            "mean_recall": 0.0

        }

    mean_dice = float(
        np.mean(
            [
                result["dice"]
                for result in valid_results
            ]
        )
    )

    mean_iou = float(
        np.mean(
            [
                result["iou"]
                for result in valid_results
            ]
        )
    )

    mean_precision = float(
        np.mean(
            [
                result["precision"]
                for result in valid_results
            ]
        )
    )

    mean_recall = float(
        np.mean(
            [
                result["recall"]
                for result in valid_results
            ]
        )
    )

    return {

        "number_of_images": len(
            valid_results
        ),

        "mean_dice": mean_dice,

        "mean_iou": mean_iou,

        "mean_precision": mean_precision,

        "mean_recall": mean_recall

    }


# ============================================================
# 16. SAVE RESULTS
# ============================================================

def save_results(
    results,
    metrics
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Chi tiet
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Tong hop metrics
    # --------------------------------------------------------

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# 17. MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "U-NET SEGMENTATION EVALUATION"
    )

    print("=" * 60)

    print()

    print(
        "Device:",
        DEVICE
    )

    print(
        "Images:",
        IMAGE_DIR
    )

    print(
        "Masks:",
        MASK_DIR
    )

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Output:",
        OUTPUT_DIR
    )

    # ========================================================
    # CHECK IMAGE DIRECTORY
    # ========================================================

    if not IMAGE_DIR.exists():

        print()

        print(
            "ERROR: Image directory not found."
        )

        print(
            "Expected:",
            IMAGE_DIR
        )

        return 1

    # ========================================================
    # CHECK MASK DIRECTORY
    # ========================================================

    if not MASK_DIR.exists():

        print()

        print(
            "ERROR: Mask directory not found."
        )

        print(
            "Expected:",
            MASK_DIR
        )

        return 1

    # ========================================================
    # CHECK MODEL
    # ========================================================

    if not MODEL_PATH.exists():

        print()

        print(
            "ERROR: U-Net model not found."
        )

        print(
            "Expected:",
            MODEL_PATH
        )

        print()

        print(
            "Hay chay Option 3 de train U-Net truoc."
        )

        return 1

    # ========================================================
    # LOAD MODEL
    # ========================================================

    try:

        model = load_model()

    except Exception as error:

        print()

        print(
            "ERROR: Cannot load U-Net model."
        )

        print(
            type(error).__name__,
            ":",
            error
        )

        return 1

    # ========================================================
    # GET IMAGES
    # ========================================================

    image_files = get_image_files()

    print()

    print(
        "Test images:",
        len(image_files)
    )

    if len(image_files) == 0:

        print()

        print(
            "ERROR: No images found."
        )

        return 1

    # ========================================================
    # EVALUATION
    # ========================================================

    results = []

    print()

    print(
        "Evaluating..."
    )

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        try:

            result = evaluate_image(
                model,
                image_path
            )

            results.append(
                result
            )

            if result.get(
                "status"
            ) == "evaluated":

                print(
                    f"[{index}/{len(image_files)}] "
                    f"{image_path.name} "
                    f"Dice={result['dice']:.4f} "
                    f"IoU={result['iou']:.4f}"
                )

            else:

                print(
                    f"[{index}/{len(image_files)}] "
                    f"{image_path.name} "
                    f"Status={result['status']}"
                )

        except Exception as error:

            print()

            print(
                f"[{index}/{len(image_files)}] "
                f"ERROR: {image_path.name}"
            )

            print(
                type(error).__name__,
                ":",
                error
            )

            results.append(
                {
                    "image": image_path.name,
                    "status": "error",
                    "error": str(error)
                }
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    metrics = calculate_summary(
        results
    )

    # ========================================================
    # PRINT RESULT
    # ========================================================

    print()

    print("=" * 60)

    print(
        "RESULT"
    )

    print("=" * 60)

    print()

    print(
        "Number of images:",
        metrics["number_of_images"]
    )

    print(
        "Mean Dice:",
        f"{metrics['mean_dice']:.4f}"
    )

    print(
        "Mean IoU:",
        f"{metrics['mean_iou']:.4f}"
    )

    print(
        "Mean Precision:",
        f"{metrics['mean_precision']:.4f}"
    )

    print(
        "Mean Recall:",
        f"{metrics['mean_recall']:.4f}"
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_results(
        results,
        metrics
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    print(
        "Segmentation results:"
    )

    print(
        OUTPUT_FILE
    )

    print()

    print(
        "Segmentation metrics:"
    )

    print(
        METRICS_FILE
    )

    print()

    print(
        "Evaluation completed."
    )

    return 0


# ============================================================
# 18. RUN
# ============================================================

if __name__ == "__main__":

    try:

        sys.exit(
            main()
        )

    except KeyboardInterrupt:

        print()

        print(
            "[WARNING] Da dung chuong trinh."
        )

        sys.exit(1)

    except Exception as error:

        print()

        print(
            "[ERROR]",
            type(error).__name__,
            ":",
            error
        )

        sys.exit(1)
