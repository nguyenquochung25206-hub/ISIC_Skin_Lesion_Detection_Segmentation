"""
<<<<<<< HEAD
U-NET SEGMENTATION EVALUATION

Danh gia mo hinh U-Net.

Input:
    data/preprocessed/images/
    data/preprocessed/masks/

Model:
    results/segmentation/best_model.pth

Output:
    results/segmentation/evaluation/
        segmentation_results.json
        segmentation_metrics.json

Metrics:
    - Dice Score
    - IoU
    - Precision
    - Recall
"""


from pathlib import Path
import json
import sys

import numpy as np

import torch
import torch.nn as nn

from PIL import Image


# ============================================================
# 1. IMPORT UNET
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


# UNet duoc dinh nghia trong train.py
from train import UNet


# ============================================================
# 2. CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


# Model
MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "segmentation"
    / "best_model.pth"
)


# Output
OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "segmentation"
    / "evaluation"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "segmentation_results.json"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "segmentation_metrics.json"
)


# ============================================================
# 3. MODEL CONFIG
# ============================================================

IMAGE_SIZE = 256

THRESHOLD = 0.5

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 4. FIND MASK
# ============================================================

def find_mask(image_path):
    """
    Tim mask tuong ung voi anh.

    Vi du:

        ISIC_0000000.jpg

    se tim:

        ISIC_0000000_Segmentation.png
        ISIC_0000000_segmentation.png
        ISIC_0000000.png
    """

    image_stem = image_path.stem

    candidates = [

        MASK_DIR
        / f"{image_stem}_Segmentation.png",

        MASK_DIR
        / f"{image_stem}_segmentation.png",

        MASK_DIR
        / f"{image_stem}.png"
    ]

    for mask_path in candidates:

        if mask_path.exists():

            return mask_path

    return None


# ============================================================
# 5. CREATE MODEL
# ============================================================

def create_model():

    model = UNet(
        in_channels=3,
        out_channels=1
    )

    return model


# ============================================================
# 6. LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Khong tim thay model: {MODEL_PATH}"
        )

    print(
        "Loading U-Net model..."
    )

    model = create_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Lay state dict
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

        elif "state_dict" in checkpoint:

            state_dict = (
                checkpoint[
                    "state_dict"
                ]
            )

        elif "model_state" in checkpoint:

            state_dict = (
                checkpoint[
                    "model_state"
                ]
            )

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

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
# 7. LOAD IMAGE
# ============================================================

def load_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        Image.Resampling.BILINEAR
    )

    image_array = np.array(
        image,
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

    tensor = torch.tensor(
        image_array,
        dtype=torch.float32
    )

    # Them batch dimension
    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.to(
        DEVICE
    )

    return tensor


# ============================================================
# 8. LOAD GROUND TRUTH MASK
# ============================================================

def load_mask(mask_path):

    mask = Image.open(
        mask_path
    ).convert("L")

    mask = mask.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        Image.Resampling.NEAREST
    )

    mask_array = np.array(
        mask,
        dtype=np.uint8
    )

    # Chuyen thanh binary
    mask_binary = (
        mask_array > 127
    ).astype(
        np.uint8
    )

    return mask_binary


# ============================================================
# 9. PREDICT MASK
# ============================================================

@torch.no_grad()
def predict_mask(
    model,
    image_tensor
):

    output = model(
        image_tensor
    )

    # Sigmoid
    probability = torch.sigmoid(
        output
    )

    probability = (
        probability
        .cpu()
        .numpy()[0, 0]
    )

    # Threshold
    predicted_mask = (
        probability >= THRESHOLD
    ).astype(
        np.uint8
    )

    return predicted_mask


# ============================================================
# 10. CALCULATE DICE
# ============================================================

def calculate_dice(
    prediction,
    ground_truth
):

    prediction = (
        prediction > 0
    )

    ground_truth = (
        ground_truth > 0
    )

    intersection = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    prediction_area = prediction.sum()

    ground_truth_area = ground_truth.sum()

    denominator = (
        prediction_area
        + ground_truth_area
    )

    if denominator == 0:

        return 1.0

    dice = (
        2.0
        * intersection
        / denominator
    )

    return float(
        dice
    )


# ============================================================
# 11. CALCULATE IOU
# ============================================================

def calculate_iou(
    prediction,
    ground_truth
):

    prediction = (
        prediction > 0
    )

    ground_truth = (
        ground_truth > 0
    )

    intersection = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    union = np.logical_or(
        prediction,
        ground_truth
    ).sum()

    if union == 0:

        return 1.0

    iou = (
        intersection
        / union
    )

    return float(
        iou
    )


# ============================================================
# 12. CALCULATE PIXEL METRICS
# ============================================================

def calculate_pixel_metrics(
    prediction,
    ground_truth
):

    prediction = (
        prediction > 0
    )

    ground_truth = (
        ground_truth > 0
    )

    # --------------------------------------------------------
    # TP
    # --------------------------------------------------------

    true_positive = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    # --------------------------------------------------------
    # FP
    # --------------------------------------------------------

    false_positive = np.logical_and(
        prediction,
        np.logical_not(
            ground_truth
        )
    ).sum()

    # --------------------------------------------------------
    # FN
    # --------------------------------------------------------

    false_negative = np.logical_and(
        np.logical_not(
            prediction
        ),
        ground_truth
    ).sum()

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = (
        true_positive
        /
        max(
            1,
            true_positive
            + false_positive
        )
    )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall = (
        true_positive
        /
        max(
            1,
            true_positive
            + false_negative
        )
    )

    return (
        int(true_positive),
        int(false_positive),
        int(false_negative),
        float(precision),
        float(recall)
    )


# ============================================================
# 13. EVALUATE ONE IMAGE
# ============================================================

def evaluate_image(
    model,
    image_path
):

    # --------------------------------------------------------
    # Find mask
    # --------------------------------------------------------

    mask_path = find_mask(
        image_path
    )

    if mask_path is None:

        return {
            "image": image_path.name,
            "status": "missing_mask"
        }

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image_tensor = load_image(
        image_path
    )

    # --------------------------------------------------------
    # Load ground truth
    # --------------------------------------------------------

    ground_truth = load_mask(
        mask_path
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = predict_mask(
        model,
        image_tensor
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    dice = calculate_dice(
        prediction,
        ground_truth
    )

    iou = calculate_iou(
        prediction,
        ground_truth
    )

    (
        true_positive,
        false_positive,
        false_negative,
        precision,
        recall
    ) = calculate_pixel_metrics(
        prediction,
        ground_truth
    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {

        "image": image_path.name,

        "mask": mask_path.name,

        "status": "evaluated",

        "dice": dice,

        "iou": iou,

        "precision": precision,

        "recall": recall,

        "true_positive": true_positive,

        "false_positive": false_positive,

        "false_negative": false_negative
    }


# ============================================================
# 14. GET IMAGE FILES
# ============================================================

def get_image_files():

    if not IMAGE_DIR.exists():

        return []

    image_files = []

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png"
    }

    # Dung iterdir de tranh trung file
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
    )


# ============================================================
# 15. CALCULATE SUMMARY
# ============================================================

def calculate_summary(
    results
):

    valid_results = [

        result

        for result in results

        if result.get("status")
        == "evaluated"

    ]

    if len(valid_results) == 0:

        return {

            "number_of_images": 0,

            "mean_dice": 0.0,

            "mean_iou": 0.0,

            "mean_precision": 0.0,

            "mean_recall": 0.0

        }

    mean_dice = float(
        np.mean(
            [
                result["dice"]
                for result in valid_results
            ]
        )
    )

    mean_iou = float(
        np.mean(
            [
                result["iou"]
                for result in valid_results
            ]
        )
    )

    mean_precision = float(
        np.mean(
            [
                result["precision"]
                for result in valid_results
            ]
        )
    )

    mean_recall = float(
        np.mean(
            [
                result["recall"]
                for result in valid_results
            ]
        )
    )

    return {

        "number_of_images": len(
            valid_results
        ),

        "mean_dice": mean_dice,

        "mean_iou": mean_iou,

        "mean_precision": mean_precision,

        "mean_recall": mean_recall

    }


# ============================================================
# 16. SAVE RESULTS
# ============================================================

def save_results(
    results,
    metrics
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Chi tiet
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Tong hop metrics
    # --------------------------------------------------------

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
# 17. MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "U-NET SEGMENTATION EVALUATION"
    )

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

    print(
        "Output:",
        OUTPUT_DIR
    )

    # ========================================================
    # CHECK IMAGE DIRECTORY
    # ========================================================

    if not IMAGE_DIR.exists():

        print()

        print(
            "ERROR: Image directory not found."
        )

        print(
            "Expected:",
            IMAGE_DIR
        )

        return 1

    # ========================================================
    # CHECK MASK DIRECTORY
    # ========================================================

    if not MASK_DIR.exists():

        print()

        print(
            "ERROR: Mask directory not found."
        )

        print(
            "Expected:",
            MASK_DIR
        )

        return 1

    # ========================================================
    # CHECK MODEL
    # ========================================================

    if not MODEL_PATH.exists():

        print()

        print(
            "ERROR: U-Net model not found."
        )

        print(
            "Expected:",
            MODEL_PATH
        )

        print()

        print(
            "Hay chay Option 3 de train U-Net truoc."
        )

        return 1

    # ========================================================
    # LOAD MODEL
    # ========================================================

    try:

        model = load_model()

    except Exception as error:

        print()

        print(
            "ERROR: Cannot load U-Net model."
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

    print()

    print(
        "Test images:",
        len(image_files)
    )

    if len(image_files) == 0:

        print()

        print(
            "ERROR: No images found."
        )

        return 1

    # ========================================================
    # EVALUATION
    # ========================================================

    results = []

    print()

    print(
        "Evaluating..."
    )

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        try:

            result = evaluate_image(
                model,
                image_path
            )

            results.append(
                result
            )

            if result.get(
                "status"
            ) == "evaluated":

                print(
                    f"[{index}/{len(image_files)}] "
                    f"{image_path.name} "
                    f"Dice={result['dice']:.4f} "
                    f"IoU={result['iou']:.4f}"
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
    # SUMMARY
    # ========================================================

    metrics = calculate_summary(
        results
    )

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

    print(
        "Number of images:",
        metrics["number_of_images"]
    )

    print(
        "Mean Dice:",
        f"{metrics['mean_dice']:.4f}"
    )

    print(
        "Mean IoU:",
        f"{metrics['mean_iou']:.4f}"
    )

    print(
        "Mean Precision:",
        f"{metrics['mean_precision']:.4f}"
    )

    print(
        "Mean Recall:",
        f"{metrics['mean_recall']:.4f}"
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_results(
        results,
        metrics
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    print(
        "Segmentation results:"
    )

    print(
        OUTPUT_FILE
    )

    print()

    print(
        "Segmentation metrics:"
    )

    print(
        METRICS_FILE
    )

    print()

    print(
        "Evaluation completed."
    )

    return 0


# ============================================================
# 18. RUN
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
[TV6] Script đánh giá U-Net đã huấn luyện trên tập test.

Cách chạy:
    python -m src.segmentation.evaluate \
        --ckpt data/output/segmentation/checkpoints/best_unet.pth \
        --data_root data/input \
        --split test \
        --out_csv results/tables/segmentation_metrics.csv

Output:
    - In ra màn hình: bảng metrics
    - Lưu CSV: results/tables/segmentation_metrics.csv
    - (Tùy chọn) Lưu ảnh trực quan overlay + compare cho từng ảnh test
"""
import os
import csv
import json
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import nội bộ
from .unet import UNet
from ..preprocessing.image_preprocessor import ISICDataset
from ..preprocessing.data_augmentation import get_val_transforms
from ..evaluation.segmentation_metrics import evaluate_batch


# ===========================================================================
# Hàm đánh giá chính: chạy vòng lặp qua toàn bộ test set
# ===========================================================================
@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    """
    Chạy model trên toàn bộ loader, trả về dict metrics trung bình.

    Chiến lược:
        - Với mỗi batch: tính metric per-sample → lấy trung bình batch đó.
        - Lưu tất cả batch vào list → mean cuối cùng.
    """
    model.eval()
    accum = {
        "dice": [], "iou": [], "pixel_acc": [],
        "precision": [], "recall": [], "specificity": [], "f1": [],
    }

    t0 = time.time()
    n_batches = len(loader)

    for i, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # Forward
        logits = model(images)              # (B,1,H,W)
        probs = torch.sigmoid(logits)       # (B,1,H,W) — xác suất [0,1]

        # Metrics cho batch này
        batch_metrics = evaluate_batch(masks, probs, threshold=threshold)
        for k, v in batch_metrics.items():
            accum[k].append(v)

        if (i + 1) % max(1, n_batches // 5) == 0 or (i + 1) == n_batches:
            print(f"  [Eval] {i+1}/{n_batches} batches "
                  f"({time.time()-t0:.1f}s) | dice={batch_metrics['dice']:.4f}")

    # Trung bình cuối cùng
    return {k: float(np.mean(v)) for k, v in accum.items()}


# ===========================================================================
# Lưu kết quả
# ===========================================================================
def save_metrics_csv(metrics, out_path):
    """Ghi dict metrics ra file CSV dạng: metric,value."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, f"{v:.6f}"])
    print(f"[Evaluate] ✅ Đã lưu CSV: {out_path}")


def save_metrics_json(metrics, out_path):
    """Ghi dict metrics ra JSON (tiện cho báo cáo)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[Evaluate] ✅ Đã lưu JSON: {out_path}")


def print_metrics(metrics, title="KẾT QUẢ ĐÁNH GIÁ SEGMENTATION"):
    """In đẹp ra console."""
    print("\n" + "=" * 50)
    print(f"  {title}")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")
    print("=" * 50 + "\n")


# ===========================================================================
# Hàm main
# ===========================================================================
def main(args):
    # ---------------- 1. Setup device ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluate] Device: {device}")

    # ---------------- 2. Load model ----------------
    model = UNet(n_channels=3, n_classes=1, bilinear=args.bilinear).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)

    # Checkpoint có thể là dict {"model_state": ...} hoặc state_dict trực tiếp
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
        print(f"[Evaluate] Checkpoint epoch={ckpt.get('epoch', '?')}, "
              f"best_dice={ckpt.get('best_dice', '?')}")
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.eval()
    print(f"[Evaluate] Đã load checkpoint: {args.ckpt}")

    # ---------------- 3. Load test dataset ----------------
    test_root = os.path.join(args.data_root, args.split)
    if not os.path.isdir(test_root):
        raise FileNotFoundError(
            f"Không tìm thấy thư mục test: {test_root}\n"
            f"Yêu cầu cấu trúc:\n"
            f"  {test_root}/\n"
            f"  ├── images/\n"
            f"  └── masks/"
        )

    test_ds = ISICDataset(
        root_dir=test_root,
        transform=get_val_transforms(image_size=(args.img_size, args.img_size)),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"[Evaluate] Số ảnh {args.split}: {len(test_ds)}")
    print(f"[Evaluate] Số batch: {len(test_loader)} | batch_size={args.batch_size}")

    # ---------------- 4. Chạy đánh giá ----------------
    metrics = evaluate(model, test_loader, device, threshold=args.threshold)

    # ---------------- 5. In kết quả ----------------
    print_metrics(metrics)

    # ---------------- 6. Lưu kết quả ----------------
    if args.out_csv:
        save_metrics_csv(metrics, args.out_csv)
    if args.out_json:
        save_metrics_json(metrics, args.out_json)

    # ---------------- 7. (Tùy chọn) Sinh ảnh trực quan ----------------
    if args.vis_dir:
        try:
            from ..segmentation.predict import Segmenter
            from ..visualization.save_results import save_all_results

            print("[Evaluate] Bắt đầu sinh ảnh trực quan...")
            seg = Segmenter(args.ckpt, img_size=args.img_size,
                            threshold=args.threshold, device=str(device))
            test_img_dir = os.path.join(test_root, "images")
            save_all_results(seg, test_img_dir, args.vis_dir)
        except Exception as e:
            print(f"[Evaluate] ⚠️ Không sinh được ảnh trực quan: {e}")

    return metrics


# ===========================================================================
# CLI
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("U-Net Evaluation (ISIC Segmentation)")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Đường dẫn checkpoint .pth")
    parser.add_argument("--data_root", type=str, default="data/input",
                        help="Thư mục gốc dataset (chứa train/val/test)")
    parser.add_argument("--split", type=str, default="test",
                        help="Tên split: train | val | test")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Ngưỡng nhị phân hóa mask dự đoán")
    parser.add_argument("--bilinear", action="store_true", default=True,
                        help="Dùng Upsample bilinear (mặc định True)")
    parser.add_argument("--out_csv", type=str,
                        default="results/tables/segmentation_metrics.csv")
    parser.add_argument("--out_json", type=str, default=None,
                        help="Nếu đặt, lưu thêm file JSON")
    parser.add_argument("--vis_dir", type=str, default=None,
                        help="Nếu đặt, sinh ảnh overlay + compare vào thư mục này")

    args = parser.parse_args()
    main(args)
>>>>>>> 84107ee48792fac379be4e497041b1766010afe1
