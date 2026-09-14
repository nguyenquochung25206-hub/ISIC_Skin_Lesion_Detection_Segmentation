<<<<<<< HEAD
"""
DU DOAN FASTER R-CNN

Input:
    data/preprocessed/images/

Model:
    results/detection/best_model.pth

Output:
    results/detection/predictions/
"""

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
from pathlib import Path
import json

import torch
<<<<<<< HEAD
from PIL import Image, ImageDraw
=======
import numpy as np
from PIL import Image, ImageDraw, ImageFont
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
from torchvision.transforms import functional as TF

from faster_rcnn import create_faster_rcnn


# ============================================================
<<<<<<< HEAD
# 1. CONFIGURATION
=======
# CONFIGURATION
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

<<<<<<< HEAD
# Anh da preprocessing
IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "preprocessed"
    / "images"
)

# Model Faster R-CNN
=======
# Anh dau vao
IMAGE_DIR = PROJECT_ROOT / "data" / "images" / "test"

# Model da train
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "detection"
    / "best_model.pth"
)

<<<<<<< HEAD
# Thu muc ket qua
=======
# Thu muc luu ket qua
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "detection"
    / "predictions"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

<<<<<<< HEAD
=======
# So class:
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
# 0 = background
# 1 = skin lesion
NUM_CLASSES = 2

<<<<<<< HEAD
# Chi giu prediction co confidence >= 0.5
SCORE_THRESHOLD = 0.5

# CPU / GPU
=======
# Nguong confidence
SCORE_THRESHOLD = 0.5

>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
<<<<<<< HEAD
# 2. LOAD MODEL
# ============================================================

def load_model():
=======
# LOAD MODEL
# ============================================================

def load_model():
    """
    Load Faster R-CNN da duoc train.
    """
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1

    print("Loading Faster R-CNN...")

    model = create_faster_rcnn(
        num_classes=NUM_CLASSES,
        pretrained=False
    )

<<<<<<< HEAD
    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Khong tim thay model: {MODEL_PATH}"
        )

=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # Ho tro nhieu dang checkpoint
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.to(DEVICE)

    model.eval()

    print("Model loaded successfully.")

    return model


# ============================================================
<<<<<<< HEAD
# 3. LOAD IMAGE
# ============================================================

def load_image(image_path):
=======
# LOAD IMAGE
# ============================================================

def load_image(image_path):
    """
    Doc anh va chuyen sang Tensor.
    """
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = TF.to_tensor(
        image
    )

    return image, image_tensor


# ============================================================
<<<<<<< HEAD
# 4. PREDICT ONE IMAGE
=======
# PREDICT ONE IMAGE
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
# ============================================================

def predict_image(
    model,
    image_tensor
):
<<<<<<< HEAD
=======
    """
    Chay Faster R-CNN tren mot anh.
    """
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1

    image_tensor = image_tensor.to(
        DEVICE
    )

    with torch.no_grad():

        prediction = model(
            [image_tensor]
        )[0]

    boxes = prediction["boxes"]
    scores = prediction["scores"]
    labels = prediction["labels"]

<<<<<<< HEAD
    # Loc confidence
=======
    # Loc theo confidence
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    keep = (
        scores
        >= SCORE_THRESHOLD
    )

    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

<<<<<<< HEAD
    return (
        boxes,
        scores,
        labels
    )


# ============================================================
# 5. DRAW BOUNDING BOX
=======
    return boxes, scores, labels


# ============================================================
# DRAW BOUNDING BOX
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
# ============================================================

def draw_predictions(
    image,
    boxes,
    scores
):
<<<<<<< HEAD
=======
    """
    Ve bounding box len anh.
    """
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1

    result_image = image.copy()

    draw = ImageDraw.Draw(
        result_image
    )

    for box, score in zip(
        boxes,
        scores
    ):

        box = (
<<<<<<< HEAD
            box
            .detach()
=======
            box.detach()
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
            .cpu()
            .numpy()
        )

        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[2])
        ymax = int(box[3])

        # Ve bounding box
        draw.rectangle(
            [
                xmin,
                ymin,
                xmax,
                ymax
            ],
            outline="red",
            width=3
        )

<<<<<<< HEAD
        # Confidence
=======
        # Noi dung confidence
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
        text = (
            f"Lesion: "
            f"{float(score):.2f}"
        )

<<<<<<< HEAD
        text_y = max(
            0,
            ymin - 20
        )

        # Background text
        text_box = draw.textbbox(
            (
                xmin,
                text_y
=======
        # Background cho text
        text_box = draw.textbbox(
            (
                xmin,
                max(0, ymin - 20)
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
            ),
            text
        )

        draw.rectangle(
            text_box,
            fill="red"
        )

        draw.text(
            (
                xmin,
<<<<<<< HEAD
                text_y
=======
                max(0, ymin - 20)
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
            ),
            text,
            fill="white"
        )

    return result_image


# ============================================================
<<<<<<< HEAD
# 6. SAVE RESULT
=======
# SAVE PREDICTION
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
# ============================================================

def save_prediction(
    image_path,
    image,
    boxes,
    scores,
    labels
):
<<<<<<< HEAD

    # --------------------------------------------------------
    # Save anh co bounding box
=======
    """
    Luu anh prediction va JSON.
    """

    # --------------------------------------------------------
    # Anh ket qua
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    # --------------------------------------------------------

    result_image = draw_predictions(
        image,
        boxes,
        scores
    )

    output_image = (
        OUTPUT_DIR
        / image_path.name
    )

    result_image.save(
        output_image
    )

    # --------------------------------------------------------
<<<<<<< HEAD
    # Save JSON
=======
    # JSON
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    # --------------------------------------------------------

    detections = []

    for box, score, label in zip(
        boxes,
        scores,
        labels
    ):

        box = (
<<<<<<< HEAD
            box
            .detach()
=======
            box.detach()
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
            .cpu()
            .numpy()
            .tolist()
        )

        detections.append(
            {
                "bbox": [
                    int(value)
                    for value in box
                ],
                "confidence": float(
<<<<<<< HEAD
                    score
                    .detach()
=======
                    score.detach()
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
                    .cpu()
                    .item()
                ),
                "class_id": int(
<<<<<<< HEAD
                    label
                    .detach()
=======
                    label.detach()
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
                    .cpu()
                    .item()
                )
            }
        )

    json_data = {
        "image": image_path.name,
        "detections": detections
    }

    output_json = (
        OUTPUT_DIR
        / f"{image_path.stem}.json"
    )

    with open(
        output_json,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            json_data,
            file,
            indent=4,
            ensure_ascii=False
        )

    return output_image


# ============================================================
<<<<<<< HEAD
# 7. GET IMAGE FILES
# ============================================================

def get_image_files():

    image_files = []

    # Duyet truc tiep, khong glob nhieu lan
    # de tranh trung lap tren Windows

    for path in IMAGE_DIR.iterdir():

        if (
            path.is_file()
            and path.suffix.lower()
            in [".jpg", ".jpeg", ".png"]
        ):

            image_files.append(path)

    return sorted(
        image_files,
        key=lambda x: x.name.lower()
=======
# GET IMAGE FILES
# ============================================================

def get_image_files():
    """
    Lay danh sach anh trong test.
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
# 8. PROCESS ONE IMAGE
=======
# PROCESS ONE IMAGE
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
# ============================================================

def process_image(
    model,
    image_path
):
<<<<<<< HEAD
=======
    """
    Xu ly mot anh.
    """
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1

    print()
    print(
        "Processing:",
        image_path.name
    )

<<<<<<< HEAD
    # Load
=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    image, image_tensor = load_image(
        image_path
    )

<<<<<<< HEAD
    # Predict
=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    boxes, scores, labels = (
        predict_image(
            model,
            image_tensor
        )
    )

    print(
        "Detections:",
        len(boxes)
    )

<<<<<<< HEAD
    # Hien thi ket qua
=======
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    for index, (
        box,
        score
    ) in enumerate(
        zip(boxes, scores),
        start=1
    ):

        box = (
<<<<<<< HEAD
            box
            .detach()
=======
            box.detach()
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
            .cpu()
            .numpy()
            .tolist()
        )

        print(
            f"  Detection {index}: "
            f"bbox={box}, "
<<<<<<< HEAD
            f"confidence="
            f"{float(score):.4f}"
        )

    # Save
=======
            f"confidence={float(score):.4f}"
        )

>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
    output_path = save_prediction(
        image_path,
        image,
        boxes,
        scores,
        labels
    )

    print(
        "Saved:",
        output_path
    )


# ============================================================
<<<<<<< HEAD
# 9. MAIN
=======
# MAIN
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
# ============================================================

def main():

    print("=" * 60)
    print(
        "FASTER R-CNN PREDICTION"
    )
    print("=" * 60)

    print()

    print(
        "Device:",
        DEVICE
    )

    print(
        "Image directory:",
        IMAGE_DIR
    )

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Output:",
        OUTPUT_DIR
    )

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        print()
        print(
            "ERROR: Model not found!"
        )

        print(
            "Expected:",
            MODEL_PATH
        )

<<<<<<< HEAD
        return 1
=======
        return
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1

    # --------------------------------------------------------
    # Check image directory
    # --------------------------------------------------------

    if not IMAGE_DIR.exists():

        print()
        print(
            "ERROR: Image directory not found!"
        )

        print(
            "Expected:",
            IMAGE_DIR
        )

<<<<<<< HEAD
        return 1
=======
        return
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

<<<<<<< HEAD
    try:

        model = load_model()

    except Exception as error:

        print()
        print(
            "ERROR: Cannot load model!"
        )

        print(
            type(error).__name__,
            ":",
            error
        )

        return 1
=======
    model = load_model()
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1

    # --------------------------------------------------------
    # Get images
    # --------------------------------------------------------

    image_files = get_image_files()

    print()

    print(
        "Number of images:",
        len(image_files)
    )

    if len(image_files) == 0:

        print(
<<<<<<< HEAD
            "ERROR: No images found."
        )

        return 1
=======
            "No images found."
        )

        return
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

<<<<<<< HEAD
    success_count = 0
    error_count = 0

    for image_path in image_files:

        try:

            process_image(
                model,
                image_path
            )

            success_count += 1

        except Exception as error:

            error_count += 1

            print()

            print(
                "ERROR:",
                image_path.name
            )

            print(
                type(error).__name__,
                ":",
                error
            )
=======
    for image_path in image_files:

        process_image(
            model,
            image_path
        )
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        "PREDICTION COMPLETED"
    )
    print("=" * 60)

    print(
<<<<<<< HEAD
        "Tong so anh:",
        len(image_files)
    )

    print(
        "Thanh cong:",
        success_count
    )

    print(
        "Loi:",
        error_count
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        OUTPUT_DIR
    )

    if error_count > 0:

        return 1

    return 0


# ============================================================
# 10. RUN
# ============================================================

if __name__ == "__main__":

    import sys

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
        "Results saved to:",
        OUTPUT_DIR
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
