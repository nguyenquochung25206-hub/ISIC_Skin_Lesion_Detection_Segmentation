<<<<<<< HEAD
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

=======
from pathlib import Path
import json

import numpy as np
import torch
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as TF


# ============================================================
<<<<<<< HEAD
# 1. CONFIG
=======
# CONFIG
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

<<<<<<< HEAD
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
=======
IMAGE_DIR = PROJECT_ROOT / "data" / "images" / "test"
MASK_DIR = PROJECT_ROOT / "data" / "masks" / "test"

MODEL_PATH = PROJECT_ROOT / "results" / "detection" / "best_model.pth"

OUTPUT_DIR = PROJECT_ROOT / "results" / "detection" / "evaluation"

OUTPUT_FILE = OUTPUT_DIR / "detection_results.json"
METRICS_FILE = OUTPUT_DIR / "detection_metrics.json"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_CLASSES = 2

SCORE_THRESHOLD = 0.5

>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
IOU_THRESHOLD = 0.5


# ============================================================
<<<<<<< HEAD
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
=======
# BOUNDING BOX FROM MASK
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
# ============================================================

def mask_to_bbox(mask_path):
    """
    Chuyen segmentation mask thanh bounding box.

<<<<<<< HEAD
    Return:
=======
    Tra ve:
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
        [xmin, ymin, xmax, ymax]

    Neu mask rong:
        None
    """

    mask = np.array(
        Image.open(mask_path).convert("L")
    )

<<<<<<< HEAD
    # Lay pixel lesion
    ys, xs = np.where(
        mask > 0
    )

    # Mask rong
    if len(xs) == 0 or len(ys) == 0:

=======
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
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
<<<<<<< HEAD
# 4. CALCULATE IOU
=======
# IOU
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
# ============================================================

def calculate_iou(box1, box2):
    """
    Tinh Intersection over Union.
<<<<<<< HEAD

    IoU = Intersection / Union
    """

    # --------------------------------------------------------
    # Intersection
    # --------------------------------------------------------

=======
    """

>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
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

<<<<<<< HEAD
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

=======
    area1 = (
        max(0, box1[2] - box1[0])
        *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0])
        *
        max(0, box2[3] - box2[1])
    )

>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    union_area = (
        area1
        + area2
        - intersection_area
    )

    if union_area <= 0:
<<<<<<< HEAD

        return 0.0

    return float(
=======
        return 0.0

    return (
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
        intersection_area
        / union_area
    )


# ============================================================
<<<<<<< HEAD
# 5. CREATE FASTER R-CNN
=======
# CREATE MODEL
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
# ============================================================

def create_model():
    """
<<<<<<< HEAD
    Tao Faster R-CNN voi 2 classes:

        0 = background
        1 = lesion
=======
    Tao Faster R-CNN.
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    """

    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=NUM_CLASSES
    )

    return model


# ============================================================
<<<<<<< HEAD
# 6. LOAD MODEL
=======
# LOAD MODEL
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
# ============================================================

def load_model():
    """
<<<<<<< HEAD
    Load Faster R-CNN da huan luyen.
    """

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Khong tim thay model: {MODEL_PATH}"
        )

    print(
        "Loading model..."
    )

=======
    Load Faster R-CNN da train.
    """

>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    model = create_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

<<<<<<< HEAD
    # --------------------------------------------------------
    # Lay state_dict
    # --------------------------------------------------------

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

<<<<<<< HEAD
        elif "model_state" in checkpoint:

            state_dict = checkpoint[
                "model_state"
            ]

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

<<<<<<< HEAD
    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    model.load_state_dict(
        state_dict
    )

<<<<<<< HEAD
    model.to(
        DEVICE
    )

    model.eval()

    print(
        "Model loaded successfully."
    )

=======
    model.to(DEVICE)

    model.eval()

>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    return model


# ============================================================
<<<<<<< HEAD
# 7. GET IMAGE FILES
=======
# GET IMAGE FILES
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
# ============================================================

def get_image_files():
    """
<<<<<<< HEAD
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
=======
    Lay tat ca anh trong test dataset.
    """

    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.JPG",
        "*.JPEG",
        "*.PNG"
    ]

    image_files = []

    for extension in extensions:

        image_files.extend(
            IMAGE_DIR.glob(extension)
        )

    return sorted(
        image_files
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    )


# ============================================================
<<<<<<< HEAD
# 8. EVALUATE ONE IMAGE
=======
# EVALUATE ONE IMAGE
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
# ============================================================

def evaluate_image(
    model,
    image_path
):
    """
    Danh gia Faster R-CNN tren mot anh.
    """

<<<<<<< HEAD
    # --------------------------------------------------------
    # Tim mask
    # --------------------------------------------------------

    mask_path = find_mask(
        image_path
    )

    if mask_path is None:
=======
    mask_path = (
        MASK_DIR
        / f"{image_path.stem}.png"
    )

    if not mask_path.exists():
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1

        return {
            "image": image_path.name,
            "status": "missing_mask"
        }

<<<<<<< HEAD
    # --------------------------------------------------------
    # Ground truth bounding box
    # --------------------------------------------------------

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    ground_truth_bbox = mask_to_bbox(
        mask_path
    )

    if ground_truth_bbox is None:

        return {
            "image": image_path.name,
            "status": "empty_mask"
        }

<<<<<<< HEAD
    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = TF.to_tensor(
        image
    )

    image_tensor = image_tensor.to(
        DEVICE
    )

<<<<<<< HEAD
    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    with torch.no_grad():

        prediction = model(
            [image_tensor]
        )[0]

<<<<<<< HEAD
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
=======
    boxes = prediction["boxes"]

    scores = prediction["scores"]

    # --------------------------------------------------------
    # Filter confidence score
    # --------------------------------------------------------

    keep = (
        scores
        >= SCORE_THRESHOLD
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
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
<<<<<<< HEAD
            "ground_truth_bbox": ground_truth_bbox,
            "predicted_bbox": None,
            "confidence": 0.0,
            "iou": 0.0,
=======
            "iou": 0.0,
            "score": 0.0,
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 1
        }

    # --------------------------------------------------------
<<<<<<< HEAD
    # Chon prediction co confidence cao nhat
=======
    # Best prediction
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
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
<<<<<<< HEAD
    # Calculate IoU
=======
    # IoU
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
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

<<<<<<< HEAD
    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
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
<<<<<<< HEAD
# 9. CALCULATE METRICS
=======
# CALCULATE METRICS
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
# ============================================================

def calculate_metrics(results):
    """
<<<<<<< HEAD
    Tinh:

        Precision
        Recall
        F1-score
        Mean IoU
=======
    Tinh Precision, Recall va Mean IoU.
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    """

    valid_results = [
        result
        for result in results
        if "iou" in result
    ]

<<<<<<< HEAD
    # Khong co ket qua hop le
=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    if len(valid_results) == 0:

        return {
            "number_of_images": 0,
            "mean_iou": 0.0,
            "precision": 0.0,
<<<<<<< HEAD
            "recall": 0.0,
            "f1_score": 0.0,
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0
        }

    # --------------------------------------------------------
    # TP / FP / FN
    # --------------------------------------------------------

=======
            "recall": 0.0
        }

>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
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

<<<<<<< HEAD
    # --------------------------------------------------------
    # Mean IoU
    # --------------------------------------------------------

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    mean_iou = float(
        np.mean(
            [
                result["iou"]
                for result in valid_results
            ]
        )
    )

<<<<<<< HEAD
    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    precision = (
        true_positive
        /
        max(
            1,
            true_positive
            + false_positive
        )
    )

<<<<<<< HEAD
    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    recall = (
        true_positive
        /
        max(
            1,
            true_positive
            + false_negative
        )
    )

<<<<<<< HEAD
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

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
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
<<<<<<< HEAD
        "f1_score": float(
            f1_score
        ),
=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
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
<<<<<<< HEAD
# 10. SAVE RESULTS
=======
# SAVE RESULTS
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
# ============================================================

def save_results(
    results,
    metrics
):
    """
<<<<<<< HEAD
    Luu ket qua evaluation thanh JSON.
=======
    Luu ket qua evaluation.
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

<<<<<<< HEAD
    # --------------------------------------------------------
    # Chi tiet tung anh
    # --------------------------------------------------------

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
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

<<<<<<< HEAD
    # --------------------------------------------------------
    # Metrics tong
    # --------------------------------------------------------

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
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
<<<<<<< HEAD
# 11. MAIN
=======
# MAIN
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
# ============================================================

def main():

    print("=" * 60)
<<<<<<< HEAD
    print(
        "FASTER R-CNN DETECTION EVALUATION"
    )
=======

    print(
        "FASTER R-CNN DETECTION EVALUATION"
    )

>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
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

<<<<<<< HEAD
    # ========================================================
    # CHECK IMAGE DIRECTORY
    # ========================================================
=======
    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1

    if not IMAGE_DIR.exists():

        print()
        print(
            "ERROR: Image directory not found."
        )

<<<<<<< HEAD
        print(
            "Expected:",
            IMAGE_DIR
        )

        return 1

    # ========================================================
    # CHECK MASK DIRECTORY
    # ========================================================
=======
        return
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1

    if not MASK_DIR.exists():

        print()
        print(
            "ERROR: Mask directory not found."
        )

<<<<<<< HEAD
        print(
            "Expected:",
            MASK_DIR
        )

        return 1

    # ========================================================
    # CHECK MODEL
    # ========================================================
=======
        return
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1

    if not MODEL_PATH.exists():

        print()
        print(
            "ERROR: Model file not found."
        )

<<<<<<< HEAD
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
=======
        return

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print(
        "Loading model..."
    )

    model = load_model()

    print(
        "Model loaded."
    )

    # --------------------------------------------------------
    # Get images
    # --------------------------------------------------------

    image_files = (
        get_image_files()
    )
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1

    print()

    print(
        "Test images:",
        len(image_files)
    )

    if len(image_files) == 0:

<<<<<<< HEAD
        print()
        print(
            "ERROR: No images found."
        )

        return 1

    # ========================================================
    # EVALUATION
    # ========================================================
=======
        print(
            "No images found."
        )

        return

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1

    results = []

    print()
    print(
        "Evaluating..."
    )

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

<<<<<<< HEAD
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
=======
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
                f"IoU={result['iou']:.4f}"
            )

        else:

            print(
                f"[{index}/{len(image_files)}] "
                f"{image_path.name} "
                f"{result['status']}"
            )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1

    metrics = calculate_metrics(
        results
    )

<<<<<<< HEAD
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

=======
    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)

>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
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
<<<<<<< HEAD
        "F1-score:",
        f"{metrics['f1_score']:.4f}"
    )

    print()

    print(
=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
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

<<<<<<< HEAD
    # ========================================================
    # SAVE
    # ========================================================
=======
    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1

    save_results(
        results,
        metrics
    )

    print()
<<<<<<< HEAD

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    print(
        "Results saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print()
<<<<<<< HEAD

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

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    print(
        "Evaluation completed."
    )

<<<<<<< HEAD
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
=======

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
