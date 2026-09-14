# ISIC Skin Lesion Detection & Segmentation

## 1. Giới thiệu

Project xây dựng hệ thống xử lý ảnh y tế nhằm phát hiện
và phân đoạn tổn thương da từ ảnh thuộc bộ dữ liệu ISIC.

Hai bài toán chính:

- Detection bằng Faster R-CNN.
- Segmentation bằng U-Net.

---

## 2. Mục tiêu

- Phát hiện vùng tổn thương da.
- Phân đoạn chính xác vùng tổn thương.
- Đánh giá hiệu quả mô hình.
- Xây dựng pipeline xử lý hoàn chỉnh.

---

## 3. Dataset

Project sử dụng dữ liệu ISIC Skin Lesion Dataset.

Dữ liệu bao gồm:

- Skin lesion images.
- Segmentation masks.
- Các thông tin cần thiết phục vụ huấn luyện và đánh giá.

Chi tiết dataset được trình bày tại:

`docs/03_research/isic_dataset.md`

---

## 4. Mô hình

### Faster R-CNN

Được sử dụng cho bài toán Object Detection.

Đầu ra:

- Bounding box.
- Confidence score.

### U-Net

Được sử dụng cho bài toán Image Segmentation.

Đầu ra:

- Segmentation mask.

---

## 5. Pipeline

```text
Input Image
     ↓
Preprocessing
     ↓
 ┌─────────────┐
 │             │
 ↓             ↓
Faster R-CNN   U-Net
 ↓             ↓
Detection      Segmentation
 ↓             ↓
Bounding Box   Mask
 └──────┬──────┘
        ↓
   Evaluation
        ↓
 Visualization
        ↓
 Final Results
