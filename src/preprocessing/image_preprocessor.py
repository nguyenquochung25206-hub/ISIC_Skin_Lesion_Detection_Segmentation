"""
ISIC Skin Lesion - Image Preprocessing

Input:
    data/input/images/
    data/input/masks/

Output:
    data/preprocessed/images/
    data/preprocessed/masks/

Chuc nang:
    - Doc anh ISIC
    - Tim mask tuong ung
    - Resize anh va mask ve 256x256
    - Normalize anh ve [0, 1]
    - Luu anh va mask sau preprocessing
"""

from pathlib import Path
import cv2
import numpy as np


# ============================================================
# 1. XAC DINH PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# 2. DUONG DAN DATA
# ============================================================

INPUT_IMAGE_DIR = PROJECT_ROOT / "data" / "input" / "images"
INPUT_MASK_DIR = PROJECT_ROOT / "data" / "input" / "masks"

OUTPUT_IMAGE_DIR = PROJECT_ROOT / "data" / "preprocessed" / "images"
OUTPUT_MASK_DIR = PROJECT_ROOT / "data" / "preprocessed" / "masks"


# ============================================================
# 3. KICH THUOC CHUAN
# ============================================================

IMAGE_SIZE = (256, 256)


# ============================================================
# 4. TIM MASK TUONG UNG
# ============================================================

def find_mask(image_path):
    """
    Tim mask tuong ung voi anh ISIC.

    Vi du:

        Image:
        ISIC_0000000.jpg

        Mask:
        ISIC_0000000_Segmentation.png
    """

    image_id = image_path.stem

    possible_masks = [
        INPUT_MASK_DIR / f"{image_id}_Segmentation.png",
        INPUT_MASK_DIR / f"{image_id}_segmentation.png",
        INPUT_MASK_DIR / f"{image_id}.png",
    ]

    for mask_path in possible_masks:
        if mask_path.exists():
            return mask_path

    return None


# ============================================================
# 5. PREPROCESSING ANH
# ============================================================

def preprocess_image(image_path):
    """
    Doc anh, resize ve 256x256 va normalize.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Khong the doc anh: {image_path}"
        )

    # OpenCV doc anh theo BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize
    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_LINEAR
    )

    # Normalize [0,255] -> [0,1]
    image = image.astype(np.float32) / 255.0

    return image


# ============================================================
# 6. PREPROCESSING MASK
# ============================================================

def preprocess_mask(mask_path):
    """
    Doc mask va resize ve 256x256.

    Mask phai dung INTER_NEAREST
    de khong lam sai nhan pixel.
    """

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise ValueError(
            f"Khong the doc mask: {mask_path}"
        )

    # Resize mask
    mask = cv2.resize(
        mask,
        IMAGE_SIZE,
        interpolation=cv2.INTER_NEAREST
    )

    # Dua mask ve binary 0/255
    mask = np.where(
        mask > 127,
        255,
        0
    ).astype(np.uint8)

    return mask


# ============================================================
# 7. LUU ANH
# ============================================================

def save_image(image, output_path):
    """
    Luu anh RGB float [0,1]
    thanh JPG [0,255].
    """

    image_uint8 = (
        np.clip(image, 0, 1) * 255
    ).astype(np.uint8)

    # RGB -> BGR de OpenCV ghi dung
    image_bgr = cv2.cvtColor(
        image_uint8,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        str(output_path),
        image_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )


# ============================================================
# 8. LUU MASK
# ============================================================

def save_mask(mask, output_path):
    """
    Luu mask binary PNG.
    """

    cv2.imwrite(
        str(output_path),
        mask
    )


# ============================================================
# 9. CHAY TOAN BO PREPROCESSING
# ============================================================

def preprocess_dataset():

    print("=" * 70)
    print("ISIC DATASET PREPROCESSING")
    print("=" * 70)

    print(f"Input images : {INPUT_IMAGE_DIR}")
    print(f"Input masks  : {INPUT_MASK_DIR}")
    print(f"Output images: {OUTPUT_IMAGE_DIR}")
    print(f"Output masks : {OUTPUT_MASK_DIR}")
    print()

    # --------------------------------------------------------
    # Kiem tra thu muc
    # --------------------------------------------------------

    if not INPUT_IMAGE_DIR.exists():
        print(
            f"[ERROR] Khong tim thay thu muc images:\n"
            f"{INPUT_IMAGE_DIR}"
        )
        return False

    if not INPUT_MASK_DIR.exists():
        print(
            f"[ERROR] Khong tim thay thu muc masks:\n"
            f"{INPUT_MASK_DIR}"
        )
        return False

    # --------------------------------------------------------
    # Tao output
    # --------------------------------------------------------

    OUTPUT_IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_MASK_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Lay danh sach anh
    # --------------------------------------------------------

    image_files = []

    for extension in ["*.jpg", "*.jpeg", "*.png"]:

        image_files.extend(
            INPUT_IMAGE_DIR.glob(extension)
        )

    image_files = sorted(image_files)

    print(
        f"Tim thay {len(image_files)} anh."
    )

    if len(image_files) == 0:

        print(
            "[ERROR] Khong co anh nao trong data/input/images/"
        )

        return False

    # --------------------------------------------------------
    # Xu ly tung anh
    # --------------------------------------------------------

    success_count = 0
    missing_mask_count = 0
    error_count = 0

    for index, image_path in enumerate(image_files, start=1):

        try:

            # Tim mask
            mask_path = find_mask(image_path)

            if mask_path is None:

                print(
                    f"[WARNING] Khong tim thay mask: "
                    f"{image_path.name}"
                )

                missing_mask_count += 1
                continue

            # ----------------------------------------------
            # Xu ly image
            # ----------------------------------------------

            image = preprocess_image(
                image_path
            )

            # ----------------------------------------------
            # Xu ly mask
            # ----------------------------------------------

            mask = preprocess_mask(
                mask_path
            )

            # ----------------------------------------------
            # Tao output path
            # ----------------------------------------------

            output_image_path = (
                OUTPUT_IMAGE_DIR
                / f"{image_path.stem}.jpg"
            )

            output_mask_path = (
                OUTPUT_MASK_DIR
                / f"{image_path.stem}.png"
            )

            # ----------------------------------------------
            # Luu
            # ----------------------------------------------

            save_image(
                image,
                output_image_path
            )

            save_mask(
                mask,
                output_mask_path
            )

            success_count += 1

            # ----------------------------------------------
            # Hien thi tien do
            # ----------------------------------------------

            if (
                index <= 10
                or index % 100 == 0
                or index == len(image_files)
            ):

                print(
                    f"[{index}/{len(image_files)}] "
                    f"{image_path.name} -> OK"
                )

        except Exception as e:

            error_count += 1

            print(
                f"[ERROR] {image_path.name}: {e}"
            )

    # ========================================================
    # TONG KET
    # ========================================================

    print()
    print("=" * 70)
    print("PREPROCESSING HOAN TAT")
    print("=" * 70)

    print(
        f"Anh xu ly thanh cong : {success_count}"
    )

    print(
        f"Thieu mask           : {missing_mask_count}"
    )

    print(
        f"Loi                   : {error_count}"
    )

    print(
        f"Output images         : {OUTPUT_IMAGE_DIR}"
    )

    print(
        f"Output masks          : {OUTPUT_MASK_DIR}"
    )

    print("=" * 70)

    return success_count > 0


# ============================================================
# 10. MAIN
# ============================================================

if __name__ == "__main__":

    success = preprocess_dataset()

    if not success:

        # Tra ve ma loi cho main.py
        raise SystemExit(1)

    raise SystemExit(0)
