# System Architecture

## 1. Tổng quan kiến trúc hệ thống

Hệ thống được xây dựng nhằm thực hiện bài toán phát hiện (Detection) và phân vùng tổn thương da (Segmentation) trên ảnh da liễu từ bộ dữ liệu ISIC.

Hệ thống được chia thành nhiều module, mỗi module đảm nhận một nhiệm vụ riêng. Các module kết hợp với nhau để tạo thành một quy trình xử lý hoàn chỉnh từ ảnh đầu vào đến kết quả cuối cùng.

Kiến trúc tổng thể của hệ thống gồm các thành phần chính:

* **Input Data**: Ảnh da liễu và Ground Truth Mask.
* **Preprocessing Module**: Tiền xử lý và chuẩn hóa ảnh.
* **Detection Module**: Sử dụng Faster R-CNN để phát hiện vùng tổn thương.
* **Segmentation Module**: Sử dụng U-Net để phân vùng chính xác vùng tổn thương.
* **Evaluation Module**: Đánh giá kết quả bằng các chỉ số phù hợp.
* **Visualization Module**: Hiển thị và lưu kết quả xử lý.
* **Output**: Kết quả Detection, Segmentation và các chỉ số đánh giá.

---

## 2. Kiến trúc tổng thể

Quy trình xử lý của hệ thống được mô tả như sau:

```text
                 +----------------------+
                 |      ISIC Dataset    |
                 |  Images + GT Masks   |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 |    Preprocessing     |
                 | Resize / Normalize   |
                 | Data Preparation     |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 |   Faster R-CNN       |
                 |      Detection       |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 |   Detection Result   |
                 | Bounding Box + Score |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 |       U-Net          |
                 |     Segmentation     |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Segmentation Result  |
                 |   Predicted Mask     |
                 +----------+-----------+
                            |
                 +----------+----------+
                 |                     |
                 v                     v
       +-------------------+   +-------------------+
       |    Evaluation     |   |  Visualization    |
       | IoU / Dice / ...  |   | Display / Save    |
       +---------+---------+   +---------+---------+
                 |                       |
                 +-----------+-----------+
                             |
                             v
                  +----------------------+
                  |    Final Results     |
                  +----------------------+
```

---

## 3. Các module chính

### 3.1. Input Data

Dữ liệu đầu vào được lấy từ bộ dữ liệu ISIC.

Dữ liệu chính gồm:

* Ảnh da liễu.
* Ground Truth Mask tương ứng với vùng tổn thương.

Ảnh được lưu trong:

```text
data/input/images/
```

Ground Truth Mask được lưu trong:

```text
data/input/masks/
```

Các dữ liệu này được sử dụng làm đầu vào cho quá trình tiền xử lý và huấn luyện, đánh giá mô hình.

---

### 3.2. Preprocessing Module

Module tiền xử lý chịu trách nhiệm chuẩn bị dữ liệu trước khi đưa vào các mô hình học sâu.

Một số bước xử lý có thể bao gồm:

* Đọc ảnh.
* Resize ảnh về kích thước phù hợp.
* Chuẩn hóa giá trị pixel.
* Xử lý mask.
* Data augmentation nếu cần thiết.

Module này được triển khai trong thư mục:

```text
src/preprocessing/
```

Các file chính gồm:

```text
image_preprocessor.py
data_augmentation.py
```

Kết quả sau tiền xử lý được lưu tại:

```text
data/processed/
```

---

### 3.3. Detection Module

Detection Module sử dụng mô hình **Faster R-CNN** để phát hiện vùng tổn thương trên ảnh.

Mục tiêu của module là xác định:

* Vị trí của vùng tổn thương.
* Bounding Box của vùng tổn thương.
* Confidence Score của dự đoán.

Module Detection được đặt tại:

```text
src/detection/
```

Các thành phần chính gồm:

```text
faster_rcnn.py
train.py
predict.py
evaluate.py
```

Kết quả Detection có thể được sử dụng cho các bước xử lý tiếp theo và được trực quan hóa bằng Bounding Box.

---

### 3.4. Segmentation Module

Segmentation Module sử dụng mô hình **U-Net** để phân vùng vùng tổn thương trên ảnh.

Khác với Detection chỉ xác định vùng bằng Bounding Box, Segmentation tạo ra một **mask dự đoán** cho từng pixel.

Module được đặt tại:

```text
src/segmentation/
```

Các thành phần chính gồm:

```text
unet.py
train.py
predict.py
evaluate.py
```

Kết quả của module là:

```text
Predicted Mask
```

Predicted Mask được sử dụng để so sánh với Ground Truth Mask trong quá trình đánh giá.

---

### 3.5. Evaluation Module

Evaluation Module chịu trách nhiệm đánh giá chất lượng kết quả của mô hình.

Đối với Segmentation, các chỉ số chính được sử dụng gồm:

* **IoU (Intersection over Union)**.
* **Dice Score**.

Module được đặt tại:

```text
src/evaluation/
```

File phụ trách đánh giá segmentation:

```text
segmentation_metrics.py
```

### IoU

IoU đo mức độ chồng lấp giữa Predicted Mask và Ground Truth Mask.

Công thức:

```text
IoU = Intersection / Union
```

Giá trị IoU nằm trong khoảng:

```text
0 → 1
```

IoU càng cao thì kết quả phân vùng càng tốt.

### Dice Score

Dice Score cũng đo mức độ tương đồng giữa Predicted Mask và Ground Truth Mask.

Công thức:

```text
Dice = 2 × Intersection / (Predicted Area + Ground Truth Area)
```

Giá trị Dice nằm trong khoảng:

```text
0 → 1
```

Dice càng gần 1 thì kết quả segmentation càng chính xác.

---

## 4. Visualization Module

Visualization Module chịu trách nhiệm hiển thị và lưu các kết quả của hệ thống.

Module được đặt tại:

```text
src/visualization/
```

Các thành phần gồm:

```text
display_detection.py
display_segmentation.py
save_results.py
```

Đối với Segmentation, hệ thống có thể trực quan hóa:

* Ảnh gốc.
* Ground Truth Mask.
* Predicted Mask.
* So sánh giữa Ground Truth và Prediction.

Kết quả trực quan có thể được lưu tại:

```text
data/output/visualizations/
```

hoặc kết quả cuối cùng phục vụ báo cáo được lưu tại:

```text
results/figures/segmentation/
```

---

## 5. Output

Sau khi hoàn thành quá trình xử lý, hệ thống tạo ra nhiều loại kết quả.

### Detection Output

Bao gồm:

* Bounding Box.
* Confidence Score.
* Detection visualization.

### Segmentation Output

Bao gồm:

* Ground Truth Mask.
* Predicted Mask.
* Segmentation visualization.
* IoU.
* Dice Score.

Các kết quả cuối cùng của Segmentation có thể được lưu tại:

```text
results/
├── figures/
│   └── segmentation/
│
├── tables/
│   └── segmentation_metrics.csv
│
└── reports/
    └── final_results.md
```

---

## 6. Luồng dữ liệu giữa các module

Luồng dữ liệu của hệ thống được thực hiện theo thứ tự:

```text
ISIC Image
    ↓
Preprocessing
    ↓
Faster R-CNN
    ↓
Detection Result
    ↓
U-Net
    ↓
Predicted Mask
    ↓
Evaluation
    ↓
IoU / Dice
    ↓
Visualization
    ↓
Final Results
```

Trong đó:

1. Ảnh ISIC được đưa vào hệ thống.
2. Ảnh được tiền xử lý để phù hợp với mô hình.
3. Faster R-CNN thực hiện phát hiện vùng tổn thương.
4. Kết quả Detection cung cấp thông tin về vùng tổn thương.
5. U-Net thực hiện phân vùng vùng tổn thương.
6. Predicted Mask được tạo ra.
7. Predicted Mask được so sánh với Ground Truth Mask.
8. Hệ thống tính các chỉ số IoU và Dice.
9. Kết quả được trực quan hóa và lưu lại.
10. Các kết quả cuối cùng được sử dụng để phân tích và đưa vào báo cáo.

---

## 7. Cấu trúc thư mục liên quan

Các thành phần chính của kiến trúc hệ thống được tổ chức như sau:

```text
src/
├── preprocessing/
│   ├── image_preprocessor.py
│   └── data_augmentation.py
│
├── detection/
│   ├── faster_rcnn.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── segmentation/
│   ├── unet.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── evaluation/
│   ├── detection_metrics.py
│   └── segmentation_metrics.py
│
└── visualization/
    ├── display_detection.py
    ├── display_segmentation.py
    └── save_results.py
```

---

## 8. Nguyên tắc thiết kế

Hệ thống được tổ chức theo hướng **module**
