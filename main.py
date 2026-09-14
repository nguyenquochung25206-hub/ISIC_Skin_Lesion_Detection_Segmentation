"""
main.py
ISIC Skin Lesion Detection & Segmentation

Chuong trinh chinh dieu khien pipeline cua project.

Chuc nang:
1. Preprocessing
2. Train Faster R-CNN
3. Train U-Net
4. Predict Detection
5. Predict Segmentation
6. Evaluate Detection
7. Evaluate Segmentation
8. Chay toan bo pipeline
0. Thoat
"""

import subprocess
import sys
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# HAM CHAY FILE PYTHON
# ============================================================

def run_script(script_relative_path):
    """
    Chay mot file Python trong project.
    """

    script_path = PROJECT_ROOT / script_relative_path

    print()
    print("=" * 70)
    print(f"DANG CHAY: {script_relative_path}")
    print("=" * 70)

    # Kiem tra file co ton tai khong
    if not script_path.exists():
        print()
        print("[ERROR] Khong tim thay file:")
        print(script_path)
        return False

    # Chay file Python
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT
    )

    # Kiem tra ket qua
    if result.returncode == 0:
        print()
        print("[OK] Chay thanh cong.")
        return True

    print()
    print("[ERROR] Chay that bai.")
    print(f"Ma loi: {result.returncode}")

    return False


# ============================================================
# MENU
# ============================================================

def show_menu():
    """
    Hien thi menu chinh.
    """

    print()
    print("=" * 70)
    print("       ISIC SKIN LESION DETECTION & SEGMENTATION")
    print("=" * 70)

    print()
    print("  1. Preprocessing")
    print("  2. Train Faster R-CNN")
    print("  3. Train U-Net")
    print("  4. Predict Detection")
    print("  5. Predict Segmentation")
    print("  6. Evaluate Detection")
    print("  7. Evaluate Segmentation")
    print("  8. Run Full Pipeline")
    print("  0. Exit")

    print()
    print("=" * 70)


# ============================================================
# PREPROCESSING
# ============================================================

def preprocessing():
    """
    Chay preprocessing anh.
    """

    return run_script(
        "src/preprocessing/image_preprocessor.py"
    )


# ============================================================
# TRAIN DETECTION
# ============================================================

def train_detection():
    """
    Train Faster R-CNN.
    """

    return run_script(
        "src/detection/train.py"
    )


# ============================================================
# TRAIN SEGMENTATION
# ============================================================

def train_segmentation():
    """
    Train U-Net.
    """

    return run_script(
        "src/segmentation/train.py"
    )


# ============================================================
# PREDICT DETECTION
# ============================================================

def predict_detection():
    """
    Chay Faster R-CNN prediction.
    """

    return run_script(
        "src/detection/predict.py"
    )


# ============================================================
# PREDICT SEGMENTATION
# ============================================================

def predict_segmentation():
    """
    Chay U-Net prediction.
    """

    return run_script(
        "src/segmentation/predict.py"
    )


# ============================================================
# EVALUATE DETECTION
# ============================================================

def evaluate_detection():
    """
    Danh gia Faster R-CNN.
    """

    return run_script(
        "src/detection/evaluate.py"
    )


# ============================================================
# EVALUATE SEGMENTATION
# ============================================================

def evaluate_segmentation():
    """
    Danh gia U-Net.
    """

    return run_script(
        "src/segmentation/evaluate.py"
    )


# ============================================================
# FULL PIPELINE
# ============================================================

def run_full_pipeline():

    print()
    print("=" * 70)
    print("                 FULL PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # BUOC 1
    # --------------------------------------------------------

    print()
    print("[1/7] PREPROCESSING")

    if not preprocessing():
        print("\n[STOP] Pipeline dung tai Preprocessing.")
        return

    # --------------------------------------------------------
    # BUOC 2
    # --------------------------------------------------------

    print()
    print("[2/7] TRAIN FASTER R-CNN")

    if not train_detection():
        print("\n[STOP] Pipeline dung tai Faster R-CNN.")
        return

    # --------------------------------------------------------
    # BUOC 3
    # --------------------------------------------------------

    print()
    print("[3/7] TRAIN U-NET")

    if not train_segmentation():
        print("\n[STOP] Pipeline dung tai U-Net.")
        return

    # --------------------------------------------------------
    # BUOC 4
    # --------------------------------------------------------

    print()
    print("[4/7] PREDICT DETECTION")

    if not predict_detection():
        print("\n[STOP] Pipeline dung tai Detection Prediction.")
        return

    # --------------------------------------------------------
    # BUOC 5
    # --------------------------------------------------------

    print()
    print("[5/7] PREDICT SEGMENTATION")

    if not predict_segmentation():
        print("\n[STOP] Pipeline dung tai Segmentation Prediction.")
        return

    # --------------------------------------------------------
    # BUOC 6
    # --------------------------------------------------------

    print()
    print("[6/7] EVALUATE DETECTION")

    if not evaluate_detection():
        print("\n[STOP] Pipeline dung tai Detection Evaluation.")
        return

    # --------------------------------------------------------
    # BUOC 7
    # --------------------------------------------------------

    print()
    print("[7/7] EVALUATE SEGMENTATION")

    if not evaluate_segmentation():
        print("\n[STOP] Pipeline dung tai Segmentation Evaluation.")
        return

    # --------------------------------------------------------
    # HOAN THANH
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("              FULL PIPELINE HOAN THANH")
    print("=" * 70)

    print()
    print("Ket qua du kien nam trong thu muc:")
    print("  results/")
    print()


# ============================================================
# MAIN
# ============================================================

def main():

    while True:

        show_menu()

        choice = input("Nhap lua chon: ").strip()

        # ----------------------------------------------------
        # 0 - EXIT
        # ----------------------------------------------------

        if choice == "0":

            print()
            print("Da thoat chuong trinh.")
            break

        # ----------------------------------------------------
        # 1 - PREPROCESSING
        # ----------------------------------------------------

        elif choice == "1":

            preprocessing()

        # ----------------------------------------------------
        # 2 - TRAIN DETECTION
        # ----------------------------------------------------

        elif choice == "2":

            train_detection()

        # ----------------------------------------------------
        # 3 - TRAIN SEGMENTATION
        # ----------------------------------------------------

        elif choice == "3":

            train_segmentation()

        # ----------------------------------------------------
        # 4 - PREDICT DETECTION
        # ----------------------------------------------------

        elif choice == "4":

            predict_detection()

        # ----------------------------------------------------
        # 5 - PREDICT SEGMENTATION
        # ----------------------------------------------------

        elif choice == "5":

            predict_segmentation()

        # ----------------------------------------------------
        # 6 - EVALUATE DETECTION
        # ----------------------------------------------------

        elif choice == "6":

            evaluate_detection()

        # ----------------------------------------------------
        # 7 - EVALUATE SEGMENTATION
        # ----------------------------------------------------

        elif choice == "7":

            evaluate_segmentation()

        # ----------------------------------------------------
        # 8 - FULL PIPELINE
        # ----------------------------------------------------

        elif choice == "8":

            run_full_pipeline()

        # ----------------------------------------------------
        # INVALID
        # ----------------------------------------------------

        else:

            print()
            print("[ERROR] Lua chon khong hop le.")
            print("Vui long chon tu 0 den 8.")

        # Cho nguoi dung doc ket qua
        print()
        input("Nhan Enter de quay lai menu...")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
