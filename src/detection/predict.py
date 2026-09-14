"""
DU DOAN FASTER R-CNN

Input:
    data/preprocessed/images/

Model:
    results/detection/best_model.pth

Output:
    results/detection/predictions/
"""

from pathlib import Path
import json

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF

from faster_rcnn import create_faster_rcnn


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Anh da preprocessing
IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "preprocessed"
    / "images"
)

# Model Faster R-CNN
MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "detection"
    / "best_model.pth"
)

# Thu muc ket qua
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

# 0 = background
# 1 = skin lesion
NUM_CLASSES = 2

# Chi giu prediction co confidence >= 0.5
SCORE_THRESHOLD = 0.5

# CPU / GPU
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 2. LOAD MODEL
# ============================================================

def load_model():

    print("Loading Faster R-CNN...")

    model = create_faster_rcnn(
        num_classes=NUM_CLASSES,
        pretrained=False
    )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Khong tim thay model: {MODEL_PATH}"
        )

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
# 3. LOAD IMAGE
# ============================================================

def load_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = TF.to_tensor(
        image
    )

    return image, image_tensor


# ============================================================
# 4. PREDICT ONE IMAGE
# ============================================================

def predict_image(
    model,
    image_tensor
):

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

    # Loc confidence
    keep = (
        scores
        >= SCORE_THRESHOLD
    )

    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    return (
        boxes,
        scores,
        labels
    )


# ============================================================
# 5. DRAW BOUNDING BOX
# ============================================================

def draw_predictions(
    image,
    boxes,
    scores
):

    result_image = image.copy()

    draw = ImageDraw.Draw(
        result_image
    )

    for box, score in zip(
        boxes,
        scores
    ):

        box = (
            box
            .detach()
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

        # Confidence
        text = (
            f"Lesion: "
            f"{float(score):.2f}"
        )

        text_y = max(
            0,
            ymin - 20
        )

        # Background text
        text_box = draw.textbbox(
            (
                xmin,
                text_y
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
                text_y
            ),
            text,
            fill="white"
        )

    return result_image


# ============================================================
# 6. SAVE RESULT
# ============================================================

def save_prediction(
    image_path,
    image,
    boxes,
    scores,
    labels
):

    # --------------------------------------------------------
    # Save anh co bounding box
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
    # Save JSON
    # --------------------------------------------------------

    detections = []

    for box, score, label in zip(
        boxes,
        scores,
        labels
    ):

        box = (
            box
            .detach()
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
                    score
                    .detach()
                    .cpu()
                    .item()
                ),
                "class_id": int(
                    label
                    .detach()
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
    )


# ============================================================
# 8. PROCESS ONE IMAGE
# ============================================================

def process_image(
    model,
    image_path
):

    print()
    print(
        "Processing:",
        image_path.name
    )

    # Load
    image, image_tensor = load_image(
        image_path
    )

    # Predict
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

    # Hien thi ket qua
    for index, (
        box,
        score
    ) in enumerate(
        zip(boxes, scores),
        start=1
    ):

        box = (
            box
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        print(
            f"  Detection {index}: "
            f"bbox={box}, "
            f"confidence="
            f"{float(score):.4f}"
        )

    # Save
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
# 9. MAIN
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

        return 1

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

        return 1

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

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
            "ERROR: No images found."
        )

        return 1

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

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
