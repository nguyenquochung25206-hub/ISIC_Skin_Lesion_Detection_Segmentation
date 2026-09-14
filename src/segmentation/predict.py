"""
<<<<<<< HEAD
DU DOAN U-NET

Input:
    data/preprocessed/images/

Model:
    results/segmentation/best_model.pth

Output:
    results/segmentation/predictions/
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

import torch


# ============================================================
# 1. IMPORT UNET
# ============================================================

# Cho phep import train.py khi file nay duoc chay truc tiep
CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


from train import UNet


# ============================================================
# 2. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Anh da preprocessing
IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "preprocessed"
    / "images"
)

# Model U-Net da train
MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "segmentation"
    / "best_model.pth"
)

# Thu muc ket qua
OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "segmentation"
    / "predictions"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. CONFIGURATION MODEL
# ============================================================

IMAGE_SIZE = 256

THRESHOLD = 0.5

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 4. LOAD MODEL
# ============================================================

def load_model():

    print(
        "Loading U-Net..."
    )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Khong tim thay model: {MODEL_PATH}"
        )

    # Tao U-Net giong voi train.py
    model = UNet(
        in_channels=3,
        out_channels=1
    )

    # Load model
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

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        "Model loaded successfully."
    )

    return model


# ============================================================
# 5. LOAD IMAGE
# ============================================================

def load_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    original_width, original_height = (
        image.size
    )

    # Resize ve 256 x 256
    image_resized = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        Image.Resampling.BILINEAR
    )

    # Chuyen numpy
    image_array = np.array(
        image_resized,
        dtype=np.float32
    )

    # Normalize [0, 1]
    image_array = (
        image_array / 255.0
    )

    # HWC -> CHW
    image_array = np.transpose(
        image_array,
        (2, 0, 1)
    )

    # Tensor
    image_tensor = torch.tensor(
        image_array,
        dtype=torch.float32
    )

    # Them batch dimension
    image_tensor = image_tensor.unsqueeze(
        0
    )

    image_tensor = image_tensor.to(
        DEVICE
    )

    return (
        image,
        image_tensor,
        original_width,
        original_height
    )


# ============================================================
# 6. PREDICT MASK
# ============================================================

@torch.no_grad()
def predict_mask(
    model,
    image_tensor,
    original_width,
    original_height
):

    # Chay U-Net
    logits = model(
        image_tensor
    )

    # Sigmoid -> probability
    probabilities = torch.sigmoid(
        logits
    )

    # Tensor -> numpy
    probability_map = (
        probabilities
        .cpu()
        .numpy()[0, 0]
    )

    # Resize mask ve kich thuoc anh goc
    probability_image = Image.fromarray(
        probability_map.astype(
            np.float32
        ),
        mode="F"
    )

    probability_image = probability_image.resize(
        (
            original_width,
            original_height
        ),
        Image.Resampling.BILINEAR
    )

    probability_map = np.array(
        probability_image,
        dtype=np.float32
    )

    # Threshold
    mask = (
        probability_map >= THRESHOLD
    ).astype(
        np.uint8
    ) * 255

    return mask


# ============================================================
# 7. SAVE MASK
# ============================================================

def save_mask(
    image_path,
    mask
):

    output_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_mask.png"
    )

    mask_image = Image.fromarray(
        mask,
        mode="L"
    )

    mask_image.save(
        output_path
    )

    return output_path


# ============================================================
# 8. SAVE OVERLAY
# ============================================================

def save_overlay(
    image,
    mask,
    image_path
):

    image_array = np.array(
        image.convert("RGB")
    )

    # Tao overlay
    overlay = image_array.copy()

    lesion = mask > 0

    # Lam noi bat vung lesion
    overlay[lesion] = (
        0.5 * overlay[lesion]
        + 0.5 * np.array(
            [255, 0, 0],
            dtype=np.float32
        )
    ).astype(
        np.uint8
    )

    overlay_image = Image.fromarray(
        overlay
    )

    output_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_overlay.jpg"
    )

    overlay_image.save(
        output_path
    )

    return output_path


# ============================================================
# 9. GET IMAGE FILES
# ============================================================

def get_image_files():

    image_files = []

    # Duyet truc tiep de tranh trung lap
    for path in IMAGE_DIR.iterdir():

        if (
            path.is_file()
            and path.suffix.lower()
            in [
                ".jpg",
                ".jpeg",
                ".png"
            ]
        ):

            image_files.append(
                path
            )

    return sorted(
        image_files,
        key=lambda x: x.name.lower()
    )


# ============================================================
# 10. PROCESS ONE IMAGE
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

    # Load image
    (
        image,
        image_tensor,
        original_width,
        original_height
    ) = load_image(
        image_path
    )

    # Predict
    mask = predict_mask(
        model,
        image_tensor,
        original_width,
        original_height
    )

    # Dem pixel lesion
    lesion_pixels = int(
        np.sum(mask > 0)
    )

    total_pixels = (
        mask.shape[0]
        * mask.shape[1]
    )

    lesion_ratio = (
        lesion_pixels
        / total_pixels
        * 100
    )

    print(
        f"Lesion pixels: "
        f"{lesion_pixels}"
    )

    print(
        f"Lesion area: "
        f"{lesion_ratio:.2f}%"
    )

    # Save mask
    mask_path = save_mask(
        image_path,
        mask
    )

    # Save overlay
    overlay_path = save_overlay(
        image,
        mask,
        image_path
    )

    print(
        "Mask saved:",
        mask_path
    )

    print(
        "Overlay saved:",
        overlay_path
    )


# ============================================================
# 11. MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print(
        "U-NET PREDICTION"
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
            "ERROR: Cannot load U-Net!"
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
        "U-NET PREDICTION COMPLETED"
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
[TV5] Dự đoán mask với U-Net đã huấn luyện.
Cung cấp class Segmenter dùng cho cả CLI và module khác (VD: visualization).
"""
import os
import cv2
import numpy as np
import torch

from .unet import UNet


class Segmenter:
    """
    Wrapper cho U-Net inference.

    Ví dụ:
        seg = Segmenter("best_unet.pth", device="cuda")
        img, mask = seg.predict("path/to/image.jpg")
    """

    def __init__(self, model_path, img_size=256, threshold=0.5,
                 device="cuda", bilinear=True):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.threshold = threshold

        # Load model
        self.model = UNet(n_channels=3, n_classes=1, bilinear=bilinear).to(self.device)
        ckpt = torch.load(model_path, map_location=self.device)
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"[Segmenter] Đã load model từ {model_path} trên {self.device}")

        # Mean/Std ImageNet để normalize (giống lúc train)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess(self, image_bgr):
        """BGR -> RGB -> resize -> normalize -> tensor (1,3,H,W)."""
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size),
                                 interpolation=cv2.INTER_LINEAR)
        img_norm = img_resized.astype(np.float32) / 255.0
        img_norm = (img_norm - self.mean) / self.std
        tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)

    @torch.no_grad()
    def predict_tensor(self, image_bgr):
        """Trả về mask cùng kích thước ảnh gốc (uint8: 0/255)."""
        h, w = image_bgr.shape[:2]
        tensor = self._preprocess(image_bgr)
        logits = self.model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
        # Resize về kích thước gốc
        mask_resized = cv2.resize(probs, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_bin = (mask_resized > self.threshold).astype(np.uint8) * 255
        return mask_bin

    def predict(self, image_path):
        """
        Trả về (ảnh_gốc_RGB, mask_uint8).
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise RuntimeError(f"Không đọc được ảnh: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = self.predict_tensor(img_bgr)
        return img_rgb, mask


# ---------------------------------------------------------------------------
# CLI test nhanh
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("U-Net Predict")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--out", type=str, default="mask.png")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    seg = Segmenter(args.ckpt, device=args.device)
    img, mask = seg.predict(args.image)
    cv2.imwrite(args.out, mask)
    print(f"Đã lưu mask -> {args.out}")
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
