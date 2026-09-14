"""
Test module for Faster R-CNN detection.

Kiểm tra:
1. Tạo model Faster R-CNN.
2. Model có đúng số class.
3. Model nhận ảnh đầu vào.
4. Output có boxes, labels, scores.
5. Hàm prediction lọc confidence đúng.
"""

import sys
from pathlib import Path

import torch
from PIL import Image

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# IMPORT
# ============================================================

from detection.faster_rcnn import (
    create_faster_rcnn,
    predict,
    get_best_prediction,
)


# ============================================================
# CONFIG
# ============================================================

NUM_CLASSES = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# TEST 1 - CREATE MODEL
# ============================================================

def test_create_model():
    """
    Kiểm tra model Faster R-CNN có thể được tạo.
    """

    model = create_faster_rcnn(
        num_classes=NUM_CLASSES,
        pretrained=False
    )

    assert model is not None

    print("[PASS] test_create_model")


# ============================================================
# TEST 2 - MODEL NUMBER OF CLASSES
# ============================================================

def test_model_num_classes():
    """
    Kiểm tra model có đúng 2 class:
    0 = background
    1 = skin lesion
    """

    model = create_faster_rcnn(
        num_classes=NUM_CLASSES,
        pretrained=False
    )

    model.eval()

    # Faster R-CNN predictor
    num_classes = model.roi_heads.box_predictor.cls_score.out_features

    assert num_classes == NUM_CLASSES

    print("[PASS] test_model_num_classes")


# ============================================================
# TEST 3 - MODEL FORWARD
# ============================================================

def test_model_forward():
    """
    Kiểm tra model có thể nhận ảnh và trả về prediction.
    """

    model = create_faster_rcnn(
        num_classes=NUM_CLASSES,
        pretrained=False
    )

    model.to(DEVICE)
    model.eval()

    # Tạo ảnh giả 256x256 RGB
    image = torch.rand(
        3,
        256,
        256,
        device=DEVICE
    )

    with torch.no_grad():

        predictions = model([image])

    assert isinstance(predictions, list)

    assert len(predictions) == 1

    prediction = predictions[0]

    assert "boxes" in prediction
    assert "labels" in prediction
    assert "scores" in prediction

    print("[PASS] test_model_forward")


# ============================================================
# TEST 4 - PREDICTION OUTPUT SHAPE
# ============================================================

def test_prediction_output_shape():
    """
    Kiểm tra kích thước output của prediction.
    """

    model = create_faster_rcnn(
        num_classes=NUM_CLASSES,
        pretrained=False
    )

    model.to(DEVICE)
    model.eval()

    image = torch.rand(
        3,
        256,
        256,
        device=DEVICE
    )

    with torch.no_grad():

        predictions = model([image])

    prediction = predictions[0]

    boxes = prediction["boxes"]
    labels = prediction["labels"]
    scores = prediction["scores"]

    # boxes phải có dạng [N, 4]
    assert boxes.ndim == 2
    assert boxes.shape[1] == 4

    # Số lượng boxes, labels và scores phải bằng nhau
    assert len(boxes) == len(labels)
    assert len(boxes) == len(scores)

    print("[PASS] test_prediction_output_shape")


# ============================================================
# TEST 5 - GET BEST PREDICTION
# ============================================================

def test_get_best_prediction():
    """
    Kiểm tra hàm lấy bounding box có confidence cao nhất.
    """

    prediction = {
        "boxes": torch.tensor([
            [10.0, 20.0, 100.0, 120.0],
            [30.0, 40.0, 150.0, 160.0],
            [50.0, 60.0, 200.0, 210.0]
        ]),

        "labels": torch.tensor([
            1,
            1,
            1
        ]),

        "scores": torch.tensor([
            0.60,
            0.90,
            0.70
        ])
    }

    best_box, best_score = get_best_prediction(
        prediction
    )

    assert best_box is not None

    assert abs(best_score - 0.90) < 1e-6

    print("[PASS] test_get_best_prediction")


# ============================================================
# TEST 6 - PREDICTION CONFIDENCE FILTER
# ============================================================

def test_prediction_confidence_filter():
    """
    Kiểm tra prediction chỉ giữ các detection
    có confidence >= threshold.
    """

    model = create_faster_rcnn(
        num_classes=NUM_CLASSES,
        pretrained=False
    )

    model.to(DEVICE)
    model.eval()

    image = torch.rand(
        3,
        256,
        256,
        device=DEVICE
    )

    boxes, labels, scores = predict(
        model,
        image,
        device=DEVICE,
        score_threshold=0.5
    )

    # Tất cả confidence phải >= 0.5
    if len(scores) > 0:
        assert torch.all(scores >= 0.5)

    print("[PASS] test_prediction_confidence_filter")


# ============================================================
# TEST 7 - REAL IMAGE
# ============================================================

def test_real_image():
    """
    Nếu dataset tồn tại, kiểm tra model với một ảnh ISIC thật.
    """

    image_dir = PROJECT_ROOT / "data" / "images" / "test"

    if not image_dir.exists():
        print("[SKIP] test_real_image - test image directory not found")
        return

    image_files = [
        p for p in image_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    if not image_files:
        print("[SKIP] test_real_image - no images found")
        return

    image_path = image_files[0]

    image = Image.open(image_path).convert("RGB")

    # Resize để test nhanh
    image = image.resize((256, 256))

    image_tensor = torch.tensor(
        list(image.getdata()),
        dtype=torch.float32
    )

    image_tensor = image_tensor.reshape(
        256,
        256,
        3
    )

    image_tensor = image_tensor.permute(
        2,
        0,
        1
    )

    image_tensor = image_tensor / 255.0

    model = create_faster_rcnn(
        num_classes=NUM_CLASSES,
        pretrained=False
    )

    model.to(DEVICE)
    model.eval()

    boxes, labels, scores = predict(
        model,
        image_tensor,
        device=DEVICE,
        score_threshold=0.5
    )

    assert boxes.ndim == 2
    assert boxes.shape[1] == 4

    print(
        f"[PASS] test_real_image - {image_path.name}"
    )


# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("FASTER R-CNN DETECTION TEST")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print()

    test_create_model()

    test_model_num_classes()

    test_model_forward()

    test_prediction_output_shape()

    test_get_best_prediction()

    test_prediction_confidence_filter()

    test_real_image()

    print()
    print("=" * 60)
    print("ALL DETECTION TESTS PASSED")
    print("=" * 60)
