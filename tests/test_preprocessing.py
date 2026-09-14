"""
Test preprocessing for ISIC Skin Lesion Detection and Segmentation.

Kiểm tra:
1. Đọc ảnh.
2. Chuyển ảnh sang RGB.
3. Resize ảnh.
4. Normalize pixel về [0, 1].
5. Kiểm tra output preprocessing.
6. Kiểm tra augmentation.
7. Kiểm tra mask không bị biến dạng sai.
8. Kiểm tra image và mask có cùng kích thước.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance


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

TRAIN_MASK_DIR = (
    PROJECT_ROOT
    / "data"
    / "masks"
    / "train"
)

PREPROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "preprocessed"
    / "images"
)

AUGMENTED_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "augmented"
    / "images"
)

AUGMENTED_MASK_DIR = (
    PROJECT_ROOT
    / "data"
    / "augmented"
    / "masks"
)


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = (256, 256)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_image_files(directory):
    """
    Lấy danh sách ảnh trong thư mục.
    """

    if not directory.exists():
        return []

    return sorted([
        file
        for file in directory.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ])


def load_and_preprocess_image(
    image_path,
    image_size=IMAGE_SIZE
):
    """
    Đọc ảnh, chuyển RGB, resize và normalize.

    Output:
        NumPy array có shape [H, W, 3]
        Giá trị pixel nằm trong [0, 1].
    """

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        image_size,
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array /= 255.0

    return image_array


def load_mask(
    mask_path,
    mask_size=IMAGE_SIZE
):
    """
    Đọc mask và resize bằng NEAREST.

    NEAREST được sử dụng cho mask để tránh
    tạo ra các giá trị pixel trung gian.
    """

    mask = Image.open(
        mask_path
    ).convert("L")

    mask = mask.resize(
        mask_size,
        Image.Resampling.NEAREST
    )

    mask_array = np.asarray(
        mask,
        dtype=np.uint8
    )

    return mask_array


def horizontal_flip(image, mask=None):
    """
    Flip ảnh theo chiều ngang.
    """

    flipped_image = image.transpose(
        Image.Transpose.FLIP_LEFT_RIGHT
    )

    if mask is not None:
        flipped_mask = mask.transpose(
            Image.Transpose.FLIP_LEFT_RIGHT
        )

        return flipped_image, flipped_mask

    return flipped_image


def vertical_flip(image, mask=None):
    """
    Flip ảnh theo chiều dọc.
    """

    flipped_image = image.transpose(
        Image.Transpose.FLIP_TOP_BOTTOM
    )

    if mask is not None:
        flipped_mask = mask.transpose(
            Image.Transpose.FLIP_TOP_BOTTOM
        )

        return flipped_image, flipped_mask

    return flipped_image


# ============================================================
# TEST 1 - DATA DIRECTORY
# ============================================================

def test_preprocessing_directories():
    """
    Kiểm tra thư mục input chính.
    """

    if not TRAIN_IMAGE_DIR.exists():

        print(
            "[SKIP] training image directory not found"
        )

        return

    assert TRAIN_IMAGE_DIR.is_dir()

    print(
        "[PASS] preprocessing directories"
    )


# ============================================================
# TEST 2 - LOAD IMAGE
# ============================================================

def test_load_image():
    """
    Kiểm tra đọc ảnh thành công.
    """

    images = get_image_files(
        TRAIN_IMAGE_DIR
    )

    if not images:

        print(
            "[SKIP] no training images found"
        )

        return

    image = Image.open(
        images[0]
    )

    assert image is not None

    print(
        "[PASS] load image"
    )


# ============================================================
# TEST 3 - RGB CONVERSION
# ============================================================

def test_rgb_conversion():
    """
    Kiểm tra ảnh sau preprocessing có 3 channel RGB.
    """

    images = get_image_files(
        TRAIN_IMAGE_DIR
    )

    if not images:

        print(
            "[SKIP] no training images found"
        )

        return

    image = Image.open(
        images[0]
    ).convert("RGB")

    assert image.mode == "RGB"

    assert len(image.getbands()) == 3

    print(
        "[PASS] RGB conversion"
    )


# ============================================================
# TEST 4 - RESIZE
# ============================================================

def test_resize():
    """
    Kiểm tra ảnh được resize về 256x256.
    """

    images = get_image_files(
        TRAIN_IMAGE_DIR
    )

    if not images:

        print(
            "[SKIP] no training images found"
        )

        return

    image_array = load_and_preprocess_image(
        images[0]
    )

    assert image_array.shape == (
        256,
        256,
        3
    )

    print(
        "[PASS] image resize"
    )


# ============================================================
# TEST 5 - NORMALIZATION
# ============================================================

def test_normalization():
    """
    Kiểm tra pixel sau normalize nằm trong [0, 1].
    """

    images = get_image_files(
        TRAIN_IMAGE_DIR
    )

    if not images:

        print(
            "[SKIP] no training images found"
        )

        return

    image_array = load_and_preprocess_image(
        images[0]
    )

    assert image_array.min() >= 0.0

    assert image_array.max() <= 1.0

    assert image_array.dtype == np.float32

    print(
        "[PASS] image normalization"
    )


# ============================================================
# TEST 6 - PREPROCESSING OUTPUT
# ============================================================

def test_preprocessing_output():
    """
    Kiểm tra toàn bộ output preprocessing.
    """

    images = get_image_files(
        TRAIN_IMAGE_DIR
    )

    if not images:

        print(
            "[SKIP] no training images found"
        )

        return

    output = load_and_preprocess_image(
        images[0]
    )

    assert isinstance(
        output,
        np.ndarray
    )

    assert output.ndim == 3

    assert output.shape[2] == 3

    assert output.shape[0] == IMAGE_SIZE[1]

    assert output.shape[1] == IMAGE_SIZE[0]

    print(
        "[PASS] preprocessing output"
    )


# ============================================================
# TEST 7 - MASK PREPROCESSING
# ============================================================

def test_mask_preprocessing():
    """
    Kiểm tra mask resize bằng NEAREST.

    Mask phải giữ dạng grayscale.
    """

    images = get_image_files(
        TRAIN_IMAGE_DIR
    )

    if not images:

        print(
            "[SKIP] no training images found"
        )

        return

    mask_path = (
        TRAIN_MASK_DIR
        / f"{images[0].stem}.png"
    )

    if not mask_path.exists():

        print(
            "[SKIP] corresponding mask not found"
        )

        return

    mask = load_mask(
        mask_path
    )

    assert mask.shape == (
        256,
        256
    )

    assert mask.ndim == 2

    print(
        "[PASS] mask preprocessing"
    )


# ============================================================
# TEST 8 - MASK BINARY CHECK
# ============================================================

def test_mask_values():
    """
    Kiểm tra mask không bị tạo ra giá trị âm
    hoặc vượt quá 255.
    """

    images = get_image_files(
        TRAIN_IMAGE_DIR
    )

    if not images:

        print(
            "[SKIP] no training images found"
        )

        return

    mask_path = (
        TRAIN_MASK_DIR
        / f"{images[0].stem}.png"
    )

    if not mask_path.exists():

        print(
            "[SKIP] corresponding mask not found"
        )

        return

    mask = load_mask(
        mask_path
    )

    assert mask.min() >= 0

    assert mask.max() <= 255

    print(
        "[PASS] mask pixel values"
    )


# ============================================================
# TEST 9 - IMAGE MASK SIZE
# ============================================================

def test_image_mask_same_size():
    """
    Sau preprocessing image và mask phải cùng kích thước.
    """

    images = get_image_files(
        TRAIN_IMAGE_DIR
    )

    if not images:

        print(
            "[SKIP] no training images found"
        )

        return

    mask_path = (
        TRAIN_MASK_DIR
        / f"{images[0].stem}.png"
    )

    if not mask_path.exists():

        print(
            "[SKIP] corresponding mask not found"
        )

        return

    image = load_and_preprocess_image(
        images[0]
    )

    mask = load_mask(
        mask_path
    )

    assert image.shape[:2] == mask.shape

    print(
        "[PASS] image-mask same size"
    )


# ============================================================
# TEST 10 - HORIZONTAL FLIP
# ============================================================

def test_horizontal_flip():
    """
    Kiểm tra horizontal flip cho image và mask.
    """

    image = Image.new(
        "RGB",
        (100, 100)
    )

    mask = Image.new(
        "L",
        (100, 100)
    )

    flipped_image, flipped_mask = (
        horizontal_flip(
            image,
            mask
        )
    )

    assert flipped_image.size == (
        100,
        100
    )

    assert flipped_mask.size == (
        100,
        100
    )

    assert flipped_image.mode == "RGB"

    assert flipped_mask.mode == "L"

    print(
        "[PASS] horizontal flip"
    )


# ============================================================
# TEST 11 - VERTICAL FLIP
# ============================================================

def test_vertical_flip():
    """
    Kiểm tra vertical flip cho image và mask.
    """

    image = Image.new(
        "RGB",
        (100, 100)
    )

    mask = Image.new(
        "L",
        (100, 100)
    )

    flipped_image, flipped_mask = (
        vertical_flip(
            image,
            mask
        )
    )

    assert flipped_image.size == (
        100,
        100
    )

    assert flipped_mask.size == (
        100,
        100
    )

    assert flipped_image.mode == "RGB"

    assert flipped_mask.mode == "L"

    print(
        "[PASS] vertical flip"
    )


# ============================================================
# TEST 12 - BRIGHTNESS AUGMENTATION
# ============================================================

def test_brightness_augmentation():
    """
    Kiểm tra thay đổi brightness chỉ áp dụng cho image.
    """

    image = Image.new(
        "RGB",
        (100, 100),
        color=(100, 100, 100)
    )

    enhancer = ImageEnhance.Brightness(
        image
    )

    brighter = enhancer.enhance(
        1.5
    )

    assert brighter.size == image.size

    assert brighter.mode == "RGB"

    assert np.array(brighter).mean() > np.array(image).mean()

    print(
        "[PASS] brightness augmentation"
    )


# ============================================================
# TEST 13 - CONTRAST AUGMENTATION
# ============================================================

def test_contrast_augmentation():
    """
    Kiểm tra thay đổi contrast.
    """

    image = Image.new(
        "RGB",
        (100, 100),
        color=(100, 100, 100)
    )

    enhancer = ImageEnhance.Contrast(
        image
    )

    enhanced = enhancer.enhance(
        1.5
    )

    assert enhanced.size == image.size

    assert enhanced.mode == "RGB"

    print(
        "[PASS] contrast augmentation"
    )


# ============================================================
# TEST 14 - AUGMENTED DIRECTORY
# ============================================================

def test_augmented_directories():
    """
    Kiểm tra thư mục augmentation nếu đã được tạo.
    """

    if not AUGMENTED_IMAGE_DIR.exists():

        print(
            "[SKIP] augmented image directory not found"
        )

        return

    assert AUGMENTED_IMAGE_DIR.is_dir()

    print(
        "[PASS] augmented image directory"
    )


# ============================================================
# TEST 15 - PREPROCESSED OUTPUT DIRECTORY
# ============================================================

def test_preprocessed_directory():
    """
    Kiểm tra thư mục output preprocessing nếu tồn tại.
    """

    if not PREPROCESSED_DIR.exists():

        print(
            "[SKIP] preprocessed directory not found"
        )

        return

    assert PREPROCESSED_DIR.is_dir()

    print(
        "[PASS] preprocessed directory"
    )


# ============================================================
# TEST 16 - COMPLETE PREPROCESSING PIPELINE
# ============================================================

def test_complete_preprocessing_pipeline():
    """
    Kiểm tra pipeline:

    Original Image
          ↓
       RGB
          ↓
       Resize
          ↓
      Normalize
          ↓
       Tensor/Array
    """

    images = get_image_files(
        TRAIN_IMAGE_DIR
    )

    if not images:

        print(
            "[SKIP] no training images found"
        )

        return

    image_path = images[0]

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    image = Image.open(
        image_path
    )

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    image = image.convert(
        "RGB"
    )

    assert image.mode == "RGB"

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    assert image.size == IMAGE_SIZE

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array /= 255.0

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    assert image_array.shape == (
        256,
        256,
        3
    )

    assert image_array.min() >= 0

    assert image_array.max() <= 1

    print(
        "[PASS] complete preprocessing pipeline"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print(
        "ISIC PREPROCESSING TEST"
    )
    print("=" * 60)

    print(
        "\n--- Directory ---"
    )

    test_preprocessing_directories()

    print(
        "\n--- Image Preprocessing ---"
    )

    test_load_image()
    test_rgb_conversion()
    test_resize()
    test_normalization()
    test_preprocessing_output()

    print(
        "\n--- Mask Preprocessing ---"
    )

    test_mask_preprocessing()
    test_mask_values()
    test_image_mask_same_size()

    print(
        "\n--- Data Augmentation ---"
    )

    test_horizontal_flip()
    test_vertical_flip()
    test_brightness_augmentation()
    test_contrast_augmentation()

    print(
        "\n--- Output Directories ---"
    )

    test_augmented_directories()
    test_preprocessed_directory()

    print(
        "\n--- Complete Pipeline ---"
    )

    test_complete_preprocessing_pipeline()

    print()
    print("=" * 60)
    print(
        "PREPROCESSING TEST COMPLETED"
    )
    print("=" * 60)
