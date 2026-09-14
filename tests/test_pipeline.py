"""
Integration tests for the ISIC Skin Lesion Detection and Segmentation pipeline.

Pipeline được kiểm tra:

Image
  ↓
Preprocessing
  ↓
Faster R-CNN Detection
  ↓
Bounding Box / ROI
  ↓
U-Net Segmentation
  ↓
Evaluation Metrics

Các test trong file này chủ yếu kiểm tra:
1. Dataset có tồn tại và đọc được.
2. Image và mask có tương ứng.
3. Image preprocessing hoạt động.
4. Detection model nhận được image.
5. Segmentation model nhận được ROI.
6. Output có đúng shape.
7. Pipeline không bị lỗi ở các bước chính.

Lưu ý:
- Đây là integration test cơ bản.
- Không đánh giá độ chính xác của model.
- Không cần model đã train để chạy các test kiểm tra cấu trúc.
"""

import sys
from pathlib import Path

import numpy as np
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
# DATA PATH
# ============================================================

TRAIN_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "images"
    / "train"
)

TEST_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "images"
    / "test"
)

TRAIN_MASK_DIR = (
    PROJECT_ROOT
    / "data"
    / "masks"
    / "train"
)

TEST_MASK_DIR = (
    PROJECT_ROOT
    / "data"
    / "masks"
    / "test"
)


# ============================================================
# CONFIG
# ============================================================

NUM_CLASSES = 2

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# IMPORT MODELS
# ============================================================

from detection.faster_rcnn import (
    create_faster_rcnn,
    predict as detection_predict,
)

from segmentation.unet import (
    UNet,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_image_files(directory):
    """
    Lấy danh sách file ảnh trong thư mục.
    """

    if not directory.exists():
        return []

    extensions = {
        ".jpg",
        ".jpeg",
        ".png"
    }

    return sorted([
        file
        for file in directory.iterdir()
        if file.is_file()
        and file.suffix.lower() in extensions
    ])


def load_image_tensor(image_path, size=(256, 256)):
    """
    Đọc ảnh và chuyển thành Tensor.

    Output:
        Tensor shape = [3, H, W]
        Giá trị pixel nằm trong [0, 1].
    """

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        size,
        Image.Resampling.LANCZOS
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array /= 255.0

    tensor = torch.from_numpy(
        image_array
    )

    tensor = tensor.permute(
        2,
        0,
        1
    )

    return tensor


# ============================================================
# TEST 1 - PROJECT DIRECTORIES
# ============================================================

def test_project_directories():
    """
    Kiểm tra các thư mục chính của project.
    """

    assert (
        PROJECT_ROOT.exists()
    )

    assert (
        SRC_DIR.exists()
    )

    assert (
        (PROJECT_ROOT / "data").exists()
    )

    print(
        "[PASS] project directories"
    )


# ============================================================
# TEST 2 - DATASET
# ============================================================

def test_dataset_exists():
    """
    Kiểm tra dataset có tồn tại.
    """

    train_images = get_image_files(
        TRAIN_IMAGE_DIR
    )

    test_images = get_image_files(
        TEST_IMAGE_DIR
    )

    if not train_images and not test_images:

        print(
            "[SKIP] dataset images not found"
        )

        return

    assert (
        len(train_images)
        + len(test_images)
        > 0
    )

    print(
        "[PASS] dataset exists"
    )


# ============================================================
# TEST 3 - IMAGE MASK MATCHING
# ============================================================

def test_image_mask_matching():
    """
    Kiểm tra image và mask có cùng tên hay không.
    """

    train_images = get_image_files(
        TRAIN_IMAGE_DIR
    )

    if not train_images:

        print(
            "[SKIP] no training images"
        )

        return

    checked = 0
    missing = 0

    for image_path in train_images:

        mask_path = (
            TRAIN_MASK_DIR
            / f"{image_path.stem}.png"
        )

        if mask_path.exists():
            checked += 1
        else:
            missing += 1

    print(
        f"Matched masks: {checked}"
    )

    print(
        f"Missing masks: {missing}"
    )

    assert checked > 0

    print(
        "[PASS] image-mask matching"
    )


# ============================================================
# TEST 4 - IMAGE PREPROCESSING
# ============================================================

def test_image_preprocessing():
    """
    Kiểm tra preprocessing ảnh.

    Kiểm tra:
    - RGB
    - resize 256x256
    - normalize về [0,1]
    - Tensor shape [3,256,256]
    """

    images = get_image_files(
        TEST_IMAGE_DIR
    )

    if not images:

        images = get_image_files(
            TRAIN_IMAGE_DIR
        )

    if not images:

        print(
            "[SKIP] no images available"
        )

        return

    image_tensor = load_image_tensor(
        images[0]
    )

    assert isinstance(
        image_tensor,
        torch.Tensor
    )

    assert image_tensor.shape == (
        3,
        256,
        256
    )

    assert (
        image_tensor.min() >= 0
    )

    assert (
        image_tensor.max() <= 1
    )

    print(
        "[PASS] image preprocessing"
    )


# ============================================================
# TEST 5 - FASTER R-CNN
# ============================================================

def test_detection_pipeline():
    """
    Kiểm tra Faster R-CNN có thể xử lý một ảnh.
    """

    images = get_image_files(
        TEST_IMAGE_DIR
    )

    if not images:

        images = get_image_files(
            TRAIN_IMAGE_DIR
        )

    if not images:

        print(
            "[SKIP] no images available"
        )

        return

    image_tensor = load_image_tensor(
        images[0]
    )

    model = create_faster_rcnn(
        num_classes=NUM_CLASSES,
        pretrained=False
    )

    model.to(DEVICE)

    model.eval()

    boxes, labels, scores = detection_predict(
        model,
        image_tensor,
        device=DEVICE,
        score_threshold=0.5
    )

    assert isinstance(
        boxes,
        torch.Tensor
    )

    assert isinstance(
        labels,
        torch.Tensor
    )

    assert isinstance(
        scores,
        torch.Tensor
    )

    assert boxes.ndim == 2

    assert boxes.shape[1] == 4

    assert len(boxes) == len(labels)

    assert len(boxes) == len(scores)

    print(
        "[PASS] Faster R-CNN pipeline"
    )


# ============================================================
# TEST 6 - DETECTION BOX
# ============================================================

def test_detection_box_format():
    """
    Kiểm tra format bounding box.

    Format:
        [x_min, y_min, x_max, y_max]
    """

    images = get_image_files(
        TEST_IMAGE_DIR
    )

    if not images:

        images = get_image_files(
            TRAIN_IMAGE_DIR
        )

    if not images:

        print(
            "[SKIP] no images available"
        )

        return

    image_tensor = load_image_tensor(
        images[0]
    )

    model = create_faster_rcnn(
        num_classes=NUM_CLASSES,
        pretrained=False
    )

    model.to(DEVICE)

    model.eval()

    boxes, labels, scores = detection_predict(
        model,
        image_tensor,
        device=DEVICE,
        score_threshold=0.0
    )

    if len(boxes) == 0:

        print(
            "[PASS] detection box format - no boxes"
        )

        return

    for box in boxes:

        assert len(box) == 4

        x1, y1, x2, y2 = box.tolist()

        assert x2 >= x1

        assert y2 >= y1

    print(
        "[PASS] detection box format"
    )


# ============================================================
# TEST 7 - UNET CREATION
# ============================================================

def test_unet_creation():
    """
    Kiểm tra U-Net có thể được tạo.
    """

    model = UNet(
        in_channels=3,
        out_channels=1
    )

    assert model is not None

    print(
        "[PASS] U-Net creation"
    )


# ============================================================
# TEST 8 - UNET FORWARD
# ============================================================

def test_unet_forward():
    """
    Kiểm tra U-Net nhận ảnh và trả về segmentation mask.
    """

    model = UNet(
        in_channels=3,
        out_channels=1
    )

    model.to(DEVICE)

    model.eval()

    image = torch.rand(
        1,
        3,
        256,
        256,
        device=DEVICE
    )

    with torch.no_grad():

        output = model(
            image
        )

    assert isinstance(
        output,
        torch.Tensor
    )

    assert output.shape == (
        1,
        1,
        256,
        256
    )

    print(
        "[PASS] U-Net forward"
    )


# ============================================================
# TEST 9 - SEGMENTATION OUTPUT
# ============================================================

def test_segmentation_output():
    """
    Kiểm tra output segmentation có thể chuyển thành binary mask.
    """

    model = UNet(
        in_channels=3,
        out_channels=1
    )

    model.to(DEVICE)

    model.eval()

    image = torch.rand(
        1,
        3,
        256,
        256,
        device=DEVICE
    )

    with torch.no_grad():

        output = model(
            image
        )

    # Nếu output là logits,
    # sigmoid để chuyển thành probability.
    probability = torch.sigmoid(
        output
    )

    binary_mask = (
        probability >= 0.5
    ).float()

    assert binary_mask.shape == (
        1,
        1,
        256,
        256
    )

    unique_values = torch.unique(
        binary_mask
    )

    for value in unique_values:

        assert value.item() in [
            0.0,
            1.0
        ]

    print(
        "[PASS] segmentation output"
    )


# ============================================================
# TEST 10 - END-TO-END TENSOR FLOW
# ============================================================

def test_end_to_end_tensor_flow():
    """
    Kiểm tra luồng dữ liệu cơ bản:

    Image
      ↓
    Tensor
      ↓
    Faster R-CNN
      ↓
    U-Net
      ↓
    Segmentation Mask
    """

    images = get_image_files(
        TEST_IMAGE_DIR
    )

    if not images:

        images = get_image_files(
            TRAIN_IMAGE_DIR
        )

    if not images:

        print(
            "[SKIP] no images available"
        )

        return

    # --------------------------------------------------------
    # Step 1: Load image
    # --------------------------------------------------------

    image_tensor = load_image_tensor(
        images[0]
    )

    assert image_tensor.shape == (
        3,
        256,
        256
    )

    # --------------------------------------------------------
    # Step 2: Detection
    # --------------------------------------------------------

    detection_model = create_faster_rcnn(
        num_classes=NUM_CLASSES,
        pretrained=False
    )

    detection_model.to(DEVICE)

    detection_model.eval()

    boxes, labels, scores = detection_predict(
        detection_model,
        image_tensor,
        device=DEVICE,
        score_threshold=0.0
    )

    assert boxes.shape[1] == 4

    # --------------------------------------------------------
    # Step 3: Segmentation
    # --------------------------------------------------------

    segmentation_model = UNet(
        in_channels=3,
        out_channels=1
    )

    segmentation_model.to(DEVICE)

    segmentation_model.eval()

    input_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        segmentation_output = (
            segmentation_model(
                input_tensor
            )
        )

    assert segmentation_output.shape == (
        1,
        1,
        256,
        256
    )

    print(
        "[PASS] end-to-end tensor flow"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print(
        "ISIC END-TO-END PIPELINE TEST"
    )
    print("=" * 60)

    print(
        f"Device: {DEVICE}"
    )

    print()

    print(
        "--- Project ---"
    )

    test_project_directories()

    print(
        "\n--- Dataset ---"
    )

    test_dataset_exists()

    test_image_mask_matching()

    print(
        "\n--- Preprocessing ---"
    )

    test_image_preprocessing()

    print(
        "\n--- Detection ---"
    )

    test_detection_pipeline()

    test_detection_box_format()

    print(
        "\n--- Segmentation ---"
    )

    test_unet_creation()

    test_unet_forward()

    test_segmentation_output()

    print(
        "\n--- End-to-End ---"
    )

    test_end_to_end_tensor_flow()

    print()

    print("=" * 60)
    print(
        "PIPELINE TEST COMPLETED"
    )
    print("=" * 60)
