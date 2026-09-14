"""
Test module for U-Net segmentation.

Kiểm tra:
1. Tạo model U-Net.
2. Forward pass.
3. Input / output shape.
4. Output segmentation mask.
5. Binary mask.
6. Mask có cùng kích thước với input.
7. Model có thể chạy trên CPU/GPU.
8. Test với ảnh ISIC thật nếu dataset tồn tại.
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

IMAGE_SIZE = (256, 256)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# IMPORT U-NET
# ============================================================

from segmentation.unet import UNet


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


def load_image_tensor(
    image_path,
    size=IMAGE_SIZE
):
    """
    Đọc ảnh và chuyển thành Tensor.

    Output:
        [3, H, W]
        pixel trong [0, 1]
    """

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        size,
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(
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


def load_mask(
    mask_path,
    size=IMAGE_SIZE
):
    """
    Đọc ground-truth mask.

    Mask được resize bằng NEAREST.
    """

    mask = Image.open(
        mask_path
    ).convert("L")

    mask = mask.resize(
        size,
        Image.Resampling.NEAREST
    )

    mask_array = np.asarray(
        mask,
        dtype=np.float32
    )

    # Chuyển mask về binary [0, 1]
    mask_array = (
        mask_array > 0
    ).astype(np.float32)

    tensor = torch.from_numpy(
        mask_array
    )

    return tensor


# ============================================================
# TEST 1 - CREATE U-NET
# ============================================================

def test_create_unet():
    """
    Kiểm tra U-Net có thể được tạo.
    """

    model = UNet(
        in_channels=3,
        out_channels=1
    )

    assert model is not None

    print(
        "[PASS] test_create_unet"
    )


# ============================================================
# TEST 2 - MODEL DEVICE
# ============================================================

def test_model_device():
    """
    Kiểm tra model có thể chuyển sang CPU/GPU.
    """

    model = UNet(
        in_channels=3,
        out_channels=1
    )

    model = model.to(
        DEVICE
    )

    parameter = next(
        model.parameters()
    )

    assert parameter.device == DEVICE

    print(
        "[PASS] test_model_device"
    )


# ============================================================
# TEST 3 - FORWARD PASS
# ============================================================

def test_unet_forward():
    """
    Kiểm tra U-Net nhận input và trả về output.
    """

    model = UNet(
        in_channels=3,
        out_channels=1
    )

    model.to(DEVICE)

    model.eval()

    input_tensor = torch.rand(
        1,
        3,
        256,
        256,
        device=DEVICE
    )

    with torch.no_grad():

        output = model(
            input_tensor
        )

    assert isinstance(
        output,
        torch.Tensor
    )

    print(
        "[PASS] test_unet_forward"
    )


# ============================================================
# TEST 4 - OUTPUT SHAPE
# ============================================================

def test_output_shape():
    """
    Kiểm tra output U-Net có shape:

        [batch, 1, height, width]
    """

    model = UNet(
        in_channels=3,
        out_channels=1
    )

    model.to(DEVICE)

    model.eval()

    input_tensor = torch.rand(
        1,
        3,
        256,
        256,
        device=DEVICE
    )

    with torch.no_grad():

        output = model(
            input_tensor
        )

    assert output.shape == (
        1,
        1,
        256,
        256
    )

    print(
        "[PASS] test_output_shape"
    )


# ============================================================
# TEST 5 - OUTPUT SPATIAL SIZE
# ============================================================

def test_output_spatial_size():
    """
    Input và output phải có cùng H x W.
    """

    model = UNet(
        in_channels=3,
        out_channels=1
    )

    model.to(DEVICE)

    model.eval()

    input_tensor = torch.rand(
        1,
        3,
        256,
        256,
        device=DEVICE
    )

    with torch.no_grad():

        output = model(
            input_tensor
        )

    assert (
        output.shape[2:]
        == input_tensor.shape[2:]
    )

    print(
        "[PASS] output spatial size"
    )


# ============================================================
# TEST 6 - SIGMOID OUTPUT
# ============================================================

def test_sigmoid_output():
    """
    Kiểm tra logits sau sigmoid nằm trong [0, 1].
    """

    model = UNet(
        in_channels=3,
        out_channels=1
    )

    model.to(DEVICE)

    model.eval()

    input_tensor = torch.rand(
        1,
        3,
        256,
        256,
        device=DEVICE
    )

    with torch.no_grad():

        logits = model(
            input_tensor
        )

        probability = torch.sigmoid(
            logits
        )

    assert probability.min() >= 0

    assert probability.max() <= 1

    print(
        "[PASS] sigmoid output"
    )


# ============================================================
# TEST 7 - BINARY MASK
# ============================================================

def test_binary_mask():
    """
    Kiểm tra probability được chuyển thành
    binary segmentation mask bằng threshold 0.5.
    """

    model = UNet(
        in_channels=3,
        out_channels=1
    )

    model.to(DEVICE)

    model.eval()

    input_tensor = torch.rand(
        1,
        3,
        256,
        256,
        device=DEVICE
    )

    with torch.no_grad():

        logits = model(
            input_tensor
        )

        probability = torch.sigmoid(
            logits
        )

        binary_mask = (
            probability >= 0.5
        ).float()

    unique_values = torch.unique(
        binary_mask
    )

    for value in unique_values:

        assert value.item() in [
            0.0,
            1.0
        ]

    print(
        "[PASS] binary mask"
    )


# ============================================================
# TEST 8 - LOAD REAL IMAGE
# ============================================================

def test_real_image():
    """
    Nếu dataset tồn tại, kiểm tra U-Net
    với một ảnh ISIC thật.
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
            "[SKIP] no ISIC images found"
        )

        return

    image_path = images[0]

    image_tensor = load_image_tensor(
        image_path
    )

    assert image_tensor.shape == (
        3,
        256,
        256
    )

    model = UNet(
        in_channels=3,
        out_channels=1
    )

    model.to(DEVICE)

    model.eval()

    input_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        output = model(
            input_tensor
        )

    assert output.shape == (
        1,
        1,
        256,
        256
    )

    print(
        f"[PASS] real image: {image_path.name}"
    )


# ============================================================
# TEST 9 - IMAGE MASK COMPATIBILITY
# ============================================================

def test_image_mask_compatibility():
    """
    Kiểm tra image và mask tương ứng có thể
    được đưa về cùng kích thước.
    """

    images = get_image_files(
        TEST_IMAGE_DIR
    )

    image_dir = TEST_IMAGE_DIR
    mask_dir = TEST_MASK_DIR

    if not images:

        images = get_image_files(
            TRAIN_IMAGE_DIR
        )

        image_dir = TRAIN_IMAGE_DIR
        mask_dir = TRAIN_MASK_DIR

    if not images:

        print(
            "[SKIP] no images found"
        )

        return

    image_path = images[0]

    mask_path = (
        mask_dir
        / f"{image_path.stem}.png"
    )

    if not mask_path.exists():

        print(
            "[SKIP] corresponding mask not found"
        )

        return

    image = load_image_tensor(
        image_path
    )

    mask = load_mask(
        mask_path
    )

    assert image.shape[:2] == mask.shape

    assert image.shape == (
        3,
        256,
        256
    )

    assert mask.shape == (
        256,
        256
    )

    print(
        "[PASS] image-mask compatibility"
    )


# ============================================================
# TEST 10 - MASK OUTPUT
# ============================================================

def test_segmentation_mask_output():
    """
    Kiểm tra output cuối cùng của segmentation.
    """

    model = UNet(
        in_channels=3,
        out_channels=1
    )

    model.to(DEVICE)

    model.eval()

    input_tensor = torch.rand(
        1,
        3,
        256,
        256,
        device=DEVICE
    )

    with torch.no_grad():

        logits = model(
            input_tensor
        )

        probability = torch.sigmoid(
            logits
        )

        mask = (
            probability >= 0.5
        ).float()

    mask = mask.squeeze(
        0
    ).squeeze(
        0
    )

    assert mask.shape == (
        256,
        256
    )

    assert mask.dtype == torch.float32

    print(
        "[PASS] segmentation mask output"
    )


# ============================================================
# TEST 11 - BATCH INPUT
# ============================================================

def test_batch_input():
    """
    Kiểm tra U-Net xử lý được batch nhiều ảnh.
    """

    model = UNet(
        in_channels=3,
        out_channels=1
    )

    model.to(DEVICE)

    model.eval()

    batch = torch.rand(
        2,
        3,
        256,
        256,
        device=DEVICE
    )

    with torch.no_grad():

        output = model(
            batch
        )

    assert output.shape == (
        2,
        1,
        256,
        256
    )

    print(
        "[PASS] batch input"
    )


# ============================================================
# TEST 12 - DIFFERENT IMAGE SIZE
# ============================================================

def test_different_image_size():
    """
    Kiểm tra U-Net với kích thước ảnh khác
    nhưng vẫn phù hợp với kiến trúc.

    128x128 được sử dụng để test nhanh.
    """

    model = UNet(
        in_channels=3,
        out_channels=1
    )

    model.to(DEVICE)

    model.eval()

    input_tensor = torch.rand(
        1,
        3,
        128,
        128,
        device=DEVICE
    )

    with torch.no_grad():

        output = model(
            input_tensor
        )

    assert output.shape == (
        1,
        1,
        128,
        128
    )

    print(
        "[PASS] different image size"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print(
        "U-NET SEGMENTATION TEST"
    )
    print("=" * 60)

    print(
        f"Device: {DEVICE}"
    )

    print(
        "\n--- Model ---"
    )

    test_create_unet()
    test_model_device()

    print(
        "\n--- Forward Pass ---"
    )

    test_unet_forward()
    test_output_shape()
    test_output_spatial_size()

    print(
        "\n--- Segmentation Output ---"
    )

    test_sigmoid_output()
    test_binary_mask()
    test_segmentation_mask_output()

    print(
        "\n--- Dataset ---"
    )

    test_real_image()
    test_image_mask_compatibility()

    print(
        "\n--- Batch / Size ---"
    )

    test_batch_input()
    test_different_image_size()

    print()
    print("=" * 60)
    print(
        "ALL SEGMENTATION TESTS COMPLETED"
    )
    print("=" * 60)
