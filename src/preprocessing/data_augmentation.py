"""
Data Augmentation for ISIC Skin Lesion Dataset.

Mục đích:
- Tăng số lượng dữ liệu training.
- Giúp mô hình học được nhiều biến đổi khác nhau của ảnh.
- Đảm bảo image và segmentation mask luôn được biến đổi giống nhau.

Các phép augmentation:
- Horizontal Flip
- Vertical Flip
- Rotation 90 / 180 / 270 độ
- Random Brightness
- Random Contrast

Lưu ý:
- Chỉ augmentation dữ liệu TRAIN.
- Không augmentation TEST/VALIDATION.
- Các phép thay đổi màu chỉ áp dụng cho image,
  không áp dụng cho mask.
"""

from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageEnhance


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "data" / "images" / "train"
MASK_DIR = PROJECT_ROOT / "data" / "masks" / "train"

OUTPUT_IMAGE_DIR = (
    PROJECT_ROOT / "data" / "augmented" / "images"
)

OUTPUT_MASK_DIR = (
    PROJECT_ROOT / "data" / "augmented" / "masks"
)

# Số phiên bản augmentation tạo thêm cho mỗi ảnh
AUGMENTATIONS_PER_IMAGE = 2

# Tỉ lệ áp dụng các phép augmentation
FLIP_PROBABILITY = 0.5
ROTATION_PROBABILITY = 0.5
BRIGHTNESS_PROBABILITY = 0.3
CONTRAST_PROBABILITY = 0.3

# Seed để kết quả có thể lặp lại
RANDOM_SEED = 42


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# AUGMENT IMAGE + MASK
# ============================================================

def augment_image_and_mask(
    image: Image.Image,
    mask: Image.Image
):
    """
    Augment image và mask.

    Image và mask phải được biến đổi hình học
    giống nhau để giữ đúng vị trí tổn thương.

    Parameters
    ----------
    image : PIL.Image
        Ảnh dermoscopic.

    mask : PIL.Image
        Mask segmentation.

    Returns
    -------
    image : PIL.Image
        Ảnh sau augmentation.

    mask : PIL.Image
        Mask sau augmentation.
    """

    # --------------------------------------------------------
    # 1. Horizontal Flip
    # --------------------------------------------------------

    if random.random() < FLIP_PROBABILITY:

        image = image.transpose(
            Image.Transpose.FLIP_LEFT_RIGHT
        )

        mask = mask.transpose(
            Image.Transpose.FLIP_LEFT_RIGHT
        )

    # --------------------------------------------------------
    # 2. Vertical Flip
    # --------------------------------------------------------

    if random.random() < FLIP_PROBABILITY:

        image = image.transpose(
            Image.Transpose.FLIP_TOP_BOTTOM
        )

        mask = mask.transpose(
            Image.Transpose.FLIP_TOP_BOTTOM
        )

    # --------------------------------------------------------
    # 3. Rotation
    # --------------------------------------------------------

    if random.random() < ROTATION_PROBABILITY:

        angle = random.choice(
            [90, 180, 270]
        )

        image = image.rotate(
            angle,
            expand=True
        )

        mask = mask.rotate(
            angle,
            expand=True
        )

    # --------------------------------------------------------
    # 4. Brightness
    # --------------------------------------------------------
    # Chỉ thay đổi image.
    # Không thay đổi mask.

    if random.random() < BRIGHTNESS_PROBABILITY:

        brightness_factor = random.uniform(
            0.8,
            1.2
        )

        enhancer = ImageEnhance.Brightness(image)

        image = enhancer.enhance(
            brightness_factor
        )

    # --------------------------------------------------------
    # 5. Contrast
    # --------------------------------------------------------

    if random.random() < CONTRAST_PROBABILITY:

        contrast_factor = random.uniform(
            0.8,
            1.2
        )

        enhancer = ImageEnhance.Contrast(image)

        image = enhancer.enhance(
            contrast_factor
        )

    return image, mask


# ============================================================
# FIND IMAGE-MASK PAIRS
# ============================================================

def find_image_mask_pairs():
    """
    Tìm các cặp image và mask tương ứng.

    Quy ước:
        image:
            data/images/train/ISIC_xxxxx.jpg

        mask:
            data/masks/train/ISIC_xxxxx.png
    """

    pairs = []

    image_extensions = [
        ".jpg",
        ".jpeg",
        ".png"
    ]

    for image_path in IMAGE_DIR.iterdir():

        if image_path.suffix.lower() not in image_extensions:
            continue

        mask_path = (
            MASK_DIR /
            f"{image_path.stem}.png"
        )

        if not mask_path.exists():
            print(
                f"[WARNING] Không tìm thấy mask: "
                f"{mask_path.name}"
            )

            continue

        pairs.append(
            (image_path, mask_path)
        )

    return pairs


# ============================================================
# SAVE AUGMENTED DATA
# ============================================================

def save_augmented_data(
    image: Image.Image,
    mask: Image.Image,
    image_name: str,
    augmentation_index: int
):
    """
    Lưu image và mask sau augmentation.
    """

    OUTPUT_IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_MASK_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    image_output_name = (
        f"{image_name}_aug_{augmentation_index}.jpg"
    )

    mask_output_name = (
        f"{image_name}_aug_{augmentation_index}.png"
    )

    image_output_path = (
        OUTPUT_IMAGE_DIR /
        image_output_name
    )

    mask_output_path = (
        OUTPUT_MASK_DIR /
        mask_output_name
    )

    # Lưu image
    image.save(
        image_output_path,
        quality=95
    )

    # Mask cần giữ dạng lossless
    # và không làm thay đổi giá trị pixel.
    mask.save(
        mask_output_path
    )


# ============================================================
# MAIN AUGMENTATION FUNCTION
# ============================================================

def run_augmentation():
    """
    Chạy augmentation cho toàn bộ training dataset.
    """

    print("=" * 60)
    print("ISIC DATA AUGMENTATION")
    print("=" * 60)

    print(f"Image directory : {IMAGE_DIR}")
    print(f"Mask directory  : {MASK_DIR}")

    pairs = find_image_mask_pairs()

    print(
        f"\nTìm thấy {len(pairs)} cặp image-mask."
    )

    if len(pairs) == 0:

        print(
            "\n[ERROR] Không tìm thấy dữ liệu."
        )

        return

    total_generated = 0

    for image_path, mask_path in pairs:

        print(
            f"\nProcessing: {image_path.name}"
        )

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            mask = Image.open(
                mask_path
            ).convert("L")

            for augmentation_index in range(
                1,
                AUGMENTATIONS_PER_IMAGE + 1
            ):

                augmented_image, augmented_mask = (
                    augment_image_and_mask(
                        image.copy(),
                        mask.copy()
                    )
                )

                save_augmented_data(
                    augmented_image,
                    augmented_mask,
                    image_path.stem,
                    augmentation_index
                )

                total_generated += 1

        except Exception as error:

            print(
                f"[ERROR] Không xử lý được "
                f"{image_path.name}: {error}"
            )

    print("\n" + "=" * 60)
    print("AUGMENTATION COMPLETED")
    print("=" * 60)

    print(
        f"Số image-mask ban đầu : {len(pairs)}"
    )

    print(
        f"Số cặp được tạo thêm  : {total_generated}"
    )

    print(
        f"Output image           : "
        f"{OUTPUT_IMAGE_DIR}"
    )

    print(
        f"Output mask            : "
        f"{OUTPUT_MASK_DIR}"
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_augmentation()
