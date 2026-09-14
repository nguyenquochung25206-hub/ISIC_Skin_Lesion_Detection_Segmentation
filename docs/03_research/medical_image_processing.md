# ISIC Dataset

## 1. Tổng quan

**ISIC (International Skin Imaging Collaboration)** là nguồn dữ liệu ảnh da liễu được sử dụng rộng rãi trong nghiên cứu **Computer Vision** và **Medical Image Analysis**.

Trong đề tài **ISIC Skin Lesion Detection and Segmentation**, dataset được sử dụng để xây dựng và đánh giá mô hình phát hiện và phân vùng tổn thương da.

---

## 2. Dữ liệu sử dụng

Dataset gồm hai thành phần chính:

```text
Image
  ↓
Ảnh dermoscopic

Mask
  ↓
Ground Truth vùng tổn thương
```

Mỗi ảnh được ghép với một segmentation mask tương ứng.

Ví dụ:

```text
ISIC_0000001.jpg
ISIC_0000001.png
```

Trong đó:

```text
.jpg → Ảnh đầu vào
.png → Ground Truth Mask
```

---

## 3. Ground Truth Mask

Mask được sử dụng làm Ground Truth cho bài toán segmentation.

Quy ước:

```text
0 → Background
1 → Skin Lesion
```

Mask giúp xác định chính xác các pixel thuộc vùng tổn thương.

Đối với bài toán Detection, từ Ground Truth Mask có thể tạo Bounding Box bao quanh vùng lesion.

```text
Ground Truth Mask
       ↓
Tìm vùng lesion
       ↓
xmin, ymin, xmax, ymax
       ↓
Bounding Box
```

---

## 4. Tiền xử lý

Trước khi đưa dữ liệu vào mô hình, ảnh được xử lý để đảm bảo cùng định dạng và kích thước.

Các bước chính:

```text
Load Image
    ↓
Resize
    ↓
Normalize
    ↓
Data Augmentation
    ↓
Model Input
```

Data Augmentation chỉ được áp dụng cho tập training.

Các phép biến đổi có thể sử dụng:

* Horizontal Flip
* Vertical Flip
* Rotation
* Random Crop
* Brightness/Contrast

---

## 5. Chia Dataset

Dữ liệu được chia thành:

```text
Train
Validation
Test
```

Trong đó:

**Train:** sử dụng để huấn luyện mô hình.

**Validation:** sử dụng để theo dõi và lựa chọn mô hình trong quá trình training.

**Test:** sử dụng để đánh giá kết quả cuối cùng.

Nếu dataset/challenge đã cung cấp sẵn cách chia tập, ưu tiên giữ nguyên cách chia đó để đảm bảo tính nhất quán.

---

## 6. Cấu trúc thư mục

Cấu trúc dữ liệu đề xuất:

```text
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
└── masks/
    ├── train/
    ├── val/
    └── test/
```

Các file ảnh và mask phải có ID tương ứng.

Ví dụ:

```text
images/train/ISIC_0000001.jpg
masks/train/ISIC_0000001.png
```

---

## 7. Sử dụng trong Project

Dataset được sử dụng cho hai nhiệm vụ:

### Detection

Sử dụng Bounding Box được tạo từ Segmentation Mask để huấn luyện và đánh giá **Faster R-CNN**.

### Segmentation

Sử dụng ảnh và Ground Truth Mask để huấn luyện mô hình segmentation, chẳng hạn **U-Net**.

Pipeline:

```text
ISIC Image
    │
    ├───────────────┐
    ▼               ▼
Detection       Segmentation
    │               │
Faster R-CNN       U-Net
    │               │
Bounding Box     Lesion Mask
```

---

## 8. Kiểm tra dữ liệu

Trước khi training cần kiểm tra:

* Ảnh có thể đọc được.
* Mask tương ứng với đúng ảnh.
* Kích thước ảnh và mask phù hợp.
* Không có file bị thiếu hoặc lỗi.
* Giá trị mask đúng định dạng.
* Dataset được chia đúng Train/Validation/Test.

---

## 9. Kết quả mong đợi

Sau khi chuẩn bị dataset, hệ thống cần đảm bảo:

```text
Image
  +
Ground Truth Mask
  ↓
Training / Validation / Testing
  ↓
Detection + Segmentation
  ↓
Evaluation
```

Dataset được chuẩn bị đúng là cơ sở để đảm bảo kết quả huấn luyện và đánh giá mô hình có tính chính xác và có thể tái lập.
