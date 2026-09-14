"""
FASTER R-CNN DETECTION EVALUATION

Danh gia mo hinh Faster R-CNN.

Input:
    data/preprocessed/images/
    data/preprocessed/masks/

Model:
    results/detection/best_model.pth

Output:
    results/detection/evaluation/
        detection_results.json
        detection_metrics.json

Metrics:
    - Mean IoU
    - Precision
    - Recall
    - True Positive
    - False Positive
    - False Negative
"""

from pathlib import Path
import json
import sys

import numpy as np
import torch

from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as TF


# ============================================================
# 1. CONFIG
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

# Model Faster R-CNN
MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "detection"
    / "best_model.pth"
)

# Output
OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "detection"
    / "evaluation"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "detection_results.json"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "detection_metrics.json"
)

# Device
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# Faster R-CNN:
# 0 = background
# 1 = lesion
NUM_CLASSES = 2

# Chi giu prediction co confidence >= 0.5
SCORE_THRESHOLD = 0.5

# IoU >= 0.5 duoc xem la detection dung
IOU_THRESHOLD = 0.5


# ============================================================
# 2. FIND MASK
# ============================================================

def find_mask(image_path):
    """
    Tim segmentation mask tuong ung voi anh.

    Vi du:

        Image:
            ISIC_0000000.jpg

        Mask:
            ISIC_0000000_Segmentation.png

    Ho tro them mot so cach dat ten khac.
    """

    image_stem = image_path.stem

    candidates = [
        MASK_DIR / f"{image_stem}_Segmentation.png",
        MASK_DIR / f"{image_stem}_segmentation.png",
        MASK_DIR / f"{image_stem}.png",
    ]

    for mask_path in candidates:

        if mask_path.exists():

            return mask_path

    return None


# ============================================================
# 3. MASK -> BOUNDING BOX
# ============================================================

def mask_to_bbox(mask_path):
    """
    Chuyen segmentation mask thanh bounding box.

    Return:
        [xmin, ymin, xmax, ymax]

    Neu mask rong:
        None
    """

    mask = np.array(
        Image.open(mask_path).convert("L")
    )

    # Lay pixel lesion
    ys, xs = np.where(
        mask > 0
    )

    # Mask rong
    if len(xs) == 0 or len(ys) == 0:

        return None

    xmin = int(xs.min())
    ymin = int(ys.min())

    xmax = int(xs.max())
    ymax = int(ys.max())

    return [
        xmin,
        ymin,
        xmax,
        ymax
    ]


# ============================================================
# 4. CALCULATE IOU
# ============================================================

def calculate_iou(box1, box2):
    """
    Tinh Intersection over Union.

    IoU = Intersection / Union
    """

    # --------------------------------------------------------
    # Intersection
    # --------------------------------------------------------

    x_left = max(
        box1[0],
        box2[0]
    )

    y_top = max(
        box1[1],
        box2[1]
    )

    x_right = min(
        box1[2],
        box2[2]
    )

    y_bottom = min(
        box1[3],
        box2[3]
    )

    intersection_width = max(
        0,
        x_right - x_left
    )

    intersection_height = max(
        0,
        y_bottom - y_top
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    # --------------------------------------------------------
    # Area box 1
    # --------------------------------------------------------

    area1 = (
        max(
            0,
            box1[2] - box1[0]
        )
        *
        max(
            0,
            box1[3] - box1[1]
        )
    )

    # --------------------------------------------------------
    # Area box 2
    # --------------------------------------------------------

    area2 = (
        max(
            0,
            box2[2] - box2[0]
        )
        *
        max(
            0,
            box2[3] - box2[1]
        )
    )

    # --------------------------------------------------------
    # Union
    # --------------------------------------------------------

    union_area = (
        area1
        + area2
        - intersection_area
    )

    if union_area <= 0:

        return 0.0

    return float(
        intersection_area
        / union_area
    )


# ============================================================
# 5. CREATE FASTER R-CNN
# ============================================================

def create_model():
    """
    Tao Faster R-CNN voi 2 classes:

        0 = background
        1 = lesion
    """

    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=NUM_CLASSES
    )

    return model


# ============================================================
# 6. LOAD MODEL
# ============================================================

def load_model():
    """
    Load Faster R-CNN da huan luyen.
    """

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Khong tim thay model: {MODEL_PATH}"
        )

    print(
        "Loading model..."
    )

    model = create_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Lay state_dict
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        elif "model_state" in checkpoint:

            state_dict = checkpoint[
                "model_state"
            ]

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

    model.to(
        DEVICE
    )

    model.eval()

    print(
        "Model loaded successfully."
    )

    return model


# ============================================================
# 7. GET IMAGE FILES
# ============================================================

def get_image_files():
    """
    Lay tat ca anh.

    Dung iterdir() thay vi glob nhieu lan
    de tranh bi trung file tren Windows.
    """

    if not IMAGE_DIR.exists():

        return []

    image_files = []

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png"
    }

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
# 8. EVALUATE ONE IMAGE
# ============================================================

def evaluate_image(
    model,
    image_path
):
    """
    Danh gia Faster R-CNN tren mot anh.
    """

    # --------------------------------------------------------
    # Tim mask
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
    # Ground truth bounding box
    # --------------------------------------------------------

    ground_truth_bbox = mask_to_bbox(
        mask_path
    )

    if ground_truth_bbox is None:

        return {
            "image": image_path.name,
            "status": "empty_mask"
        }

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = TF.to_tensor(
        image
    )

    image_tensor = image_tensor.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        prediction = model(
            [image_tensor]
        )[0]

    boxes = prediction[
        "boxes"
    ]

    scores = prediction[
        "scores"
    ]

    # --------------------------------------------------------
    # Confidence filtering
    # --------------------------------------------------------

    keep = (
        scores >= SCORE_THRESHOLD
    )

    boxes = boxes[keep]

    scores = scores[keep]

    # --------------------------------------------------------
    # No detection
    # --------------------------------------------------------

    if len(boxes) == 0:

        return {
            "image": image_path.name,
            "status": "no_detection",
            "ground_truth_bbox": ground_truth_bbox,
            "predicted_bbox": None,
            "confidence": 0.0,
            "iou": 0.0,
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 1
        }

    # --------------------------------------------------------
    # Chon prediction co confidence cao nhat
    # --------------------------------------------------------

    best_index = torch.argmax(
        scores
    )

    predicted_bbox = (
        boxes[best_index]
        .detach()
        .cpu()
        .numpy()
        .tolist()
    )

    predicted_bbox = [
        int(value)
        for value in predicted_bbox
    ]

    confidence = float(
        scores[best_index]
        .detach()
        .cpu()
        .item()
    )

    # --------------------------------------------------------
    # Calculate IoU
    # --------------------------------------------------------

    iou = calculate_iou(
        ground_truth_bbox,
        predicted_bbox
    )

    # --------------------------------------------------------
    # TP / FP / FN
    # --------------------------------------------------------

    if iou >= IOU_THRESHOLD:

        status = "correct"

        true_positive = 1
        false_positive = 0
        false_negative = 0

    else:

        status = "incorrect"

        true_positive = 0
        false_positive = 1
        false_negative = 1

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {
        "image": image_path.name,
        "status": status,
        "ground_truth_bbox": ground_truth_bbox,
        "predicted_bbox": predicted_bbox,
        "confidence": confidence,
        "iou": float(iou),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative
    }


# ============================================================
# 9. CALCULATE METRICS
# ============================================================

def calculate_metrics(results):
    """
    Tinh:

        Precision
        Recall
        F1-score
        Mean IoU
    """

    valid_results = [
        result
        for result in results
        if "iou" in result
    ]

    # Khong co ket qua hop le
    if len(valid_results) == 0:

        return {
            "number_of_images": 0,
            "mean_iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0
        }

    # --------------------------------------------------------
    # TP / FP / FN
    # --------------------------------------------------------

    true_positive = sum(
        result["true_positive"]
        for result in valid_results
    )

    false_positive = sum(
        result["false_positive"]
        for result in valid_results
    )

    false_negative = sum(
        result["false_negative"]
        for result in valid_results
    )

    # --------------------------------------------------------
    # Mean IoU
    # --------------------------------------------------------

    mean_iou = float(
        np.mean(
            [
                result["iou"]
                for result in valid_results
            ]
        )
    )

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

    # --------------------------------------------------------
    # F1-score
    # --------------------------------------------------------

    if (
        precision + recall
    ) > 0:

        f1_score = (
            2
            * precision
            * recall
            /
            (precision + recall)
        )

    else:

        f1_score = 0.0

    return {
        "number_of_images": len(
            valid_results
        ),
        "mean_iou": mean_iou,
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "f1_score": float(
            f1_score
        ),
        "true_positive": int(
            true_positive
        ),
        "false_positive": int(
            false_positive
        ),
        "false_negative": int(
            false_negative
        )
    }


# ============================================================
# 10. SAVE RESULTS
# ============================================================

def save_results(
    results,
    metrics
):
    """
    Luu ket qua evaluation thanh JSON.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Chi tiet tung anh
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
    # Metrics tong
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
# 11. MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "FASTER R-CNN DETECTION EVALUATION"
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
            "ERROR: Model file not found."
        )

        print(
            "Expected:",
            MODEL_PATH
        )

        print()
        print(
            "Hay chay Option 2 de train Faster R-CNN truoc."
        )

        return 1

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()

    try:

        model = load_model()

    except Exception as error:

        print()

        print(
            "ERROR: Cannot load model."
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

            if "iou" in result:

                print(
                    f"[{index}/{len(image_files)}] "
                    f"{image_path.name} "
                    f"IoU={result['iou']:.4f} "
                    f"Status={result['status']}"
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
    # CALCULATE METRICS
    # ========================================================

    metrics = calculate_metrics(
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
        "Mean IoU:",
        f"{metrics['mean_iou']:.4f}"
    )

    print(
        "Precision:",
        f"{metrics['precision']:.4f}"
    )

    print(
        "Recall:",
        f"{metrics['recall']:.4f}"
    )

    print(
        "F1-score:",
        f"{metrics['f1_score']:.4f}"
    )

    print()

    print(
        "True Positive:",
        metrics["true_positive"]
    )

    print(
        "False Positive:",
        metrics["false_positive"]
    )

    print(
        "False Negative:",
        metrics["false_negative"]
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
        "Detection results:"
    )

    print(
        OUTPUT_FILE
    )

    print()

    print(
        "Detection metrics:"
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
# 12. RUN
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
