# Test Plan

## 1. Mục đích

Kiểm thử hệ thống **ISIC Skin Lesion Detection & Segmentation** để đảm bảo toàn bộ pipeline hoạt động đúng từ dữ liệu đầu vào đến kết quả cuối cùng.

```text
ISIC Image
    ↓
Preprocessing
    ↓
Faster R-CNN
    ↓
Bounding Box
    ↓
Crop ROI
    ↓
U-Net
    ↓
Segmentation Mask
    ↓
Evaluation
    ↓
Results
```

---

## 2. Phạm vi kiểm thử

Kiểm thử các thành phần chính:

| Thành phần      | Nội dung                        |
| --------------- | ------------------------------- |
| Dataset         | Đọc ảnh và Ground Truth Mask    |
| Preprocessing   | Resize, Normalize, Augmentation |
| Detection       | Faster R-CNN                    |
| ROI             | Crop vùng lesion                |
| Segmentation    | U-Net                           |
| Post-processing | Threshold, xử lý mask           |
| Evaluation      | Dice, IoU, Precision, Recall    |
| Visualization   | Prediction và Overlay           |
| Output          | Lưu kết quả                     |

---

## 3. Mục tiêu kiểm thử

### 3.1. Functional Testing

Kiểm tra các chức năng của hệ thống:

* Đọc và xử lý ảnh.
* Chạy Faster R-CNN.
* Tạo Bounding Box.
* Crop ROI.
* Chạy U-Net.
* Tạo Segmentation Mask.
* Tính các evaluation metrics.
* Lưu và hiển thị kết quả.

### 3.2. Model Testing

Đánh giá khả năng của hai model:

**Faster R-CNN**

* Bounding Box.
* Confidence.
* IoU.
* Precision.
* Recall.
* mAP nếu có.

**U-Net**

* Dice.
* IoU.
* Precision.
* Recall.

---

## 4. Test Environment

| Thành phần         | Thiết lập           |
| ------------------ | ------------------- |
| OS                 | Windows             |
| Language           | Python 3.11         |
| IDE                | Visual Studio Code  |
| Dataset            | ISIC                |
| Detection Model    | Faster R-CNN        |
| Segmentation Model | U-Net               |
| Image Type         | RGB                 |
| Input              | Dermoscopic image   |
| Output             | Bounding Box + Mask |

---

## 5. Test Data

Test sử dụng các ảnh ISIC có Ground Truth Mask để có thể so sánh kết quả dự đoán.

Dữ liệu được chia:

```text
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── masks/
    ├── train/
    ├── val/
    └── test/
```

**Lưu ý:** Test set không được sử dụng trong quá trình huấn luyện model.

---

## 6. Test Strategy

### 6.1. Unit Testing

Kiểm tra từng module độc lập:

```text
Image Loading
Preprocessing
Detection
ROI Crop
Segmentation
Metrics
Visualization
```

### 6.2. Integration Testing

Kiểm tra khả năng kết nối giữa các module:

```text
Faster R-CNN
      ↓
Bounding Box
      ↓
ROI Crop
      ↓
U-Net
```

### 6.3. End-to-End Testing

Đưa một ảnh ISIC vào hệ thống và kiểm tra toàn bộ pipeline:

```text
Input
 → Detection
 → Crop
 → Segmentation
 → Evaluation
 → Output
```

---

## 7. Test Scenarios

| ID   | Scenario         | Expected Result               |
| ---- | ---------------- | ----------------------------- |
| TS01 | Input ảnh hợp lệ | Đọc thành công                |
| TS02 | Input ảnh lỗi    | Báo lỗi an toàn               |
| TS03 | Preprocessing    | Ảnh đúng kích thước/chuẩn hóa |
| TS04 | Detection        | Có Bounding Box               |
| TS05 | Crop ROI         | Cắt đúng vùng                 |
| TS06 | Segmentation     | Sinh được Mask                |
| TS07 | Post-processing  | Mask nhị phân đúng            |
| TS08 | Evaluation       | Tính được metrics             |
| TS09 | Visualization    | Hiển thị đúng kết quả         |
| TS10 | Full Pipeline    | Hoàn thành không lỗi          |

Chi tiết từng test case được mô tả trong [`test_cases.md`](test_cases.md).

---

## 8. Evaluation Criteria

Các metric chính:

| Metric    | Mục đích                                     |
| --------- | -------------------------------------------- |
| Dice      | Độ chồng lấp giữa Prediction và Ground Truth |
| IoU       | Mức độ giao nhau giữa hai vùng               |
| Precision | Mức độ chính xác của vùng dự đoán            |
| Recall    | Khả năng phát hiện đầy đủ lesion             |

Đối với Detection có thể sử dụng thêm:

```text
IoU
Precision
Recall
mAP
```

---

## 9. Pass / Fail Criteria

### PASS

Test được xem là PASS khi:

* Chức năng hoạt động đúng.
* Không xảy ra lỗi ngoài dự kiến.
* Output đúng định dạng.
* Model tạo được kết quả dự đoán.
* Metrics được tính thành công.
* Kết quả được lưu đúng thư mục.

### FAIL

Test được xem là FAIL khi:

* Chương trình bị crash.
* Không xử lý được input hợp lệ.
* Không tạo được Bounding Box.
* Không tạo được Segmentation Mask.
* Output sai kích thước hoặc định dạng.
* Không tính hoặc lưu được metrics.

---

## 10. Test Results

Sau khi hoàn thành kiểm thử, nhóm tổng hợp:

| Tổng Test | PASS | FAIL | Pass Rate |
| --------: | ---: | ---: | --------: |
|        20 |      |      |           |

Công thức:

```text
Pass Rate = PASS / Total Tests × 100%
```

---

## 11. Deliverables

Sau quá trình kiểm thử, cần có:

```text
results/
├── predictions/       # Kết quả dự đoán
├── visualizations/    # Ảnh trực quan
├── metrics/           # Dice, IoU, Precision, Recall
└── plots/             # Biểu đồ kết quả
```

Các tài liệu kiểm thử:

```text
test_plan.md
test_cases.md
test_report.md
```

---

## 12. Completion Criteria

Testing được xem là hoàn thành khi:

* Tất cả test case đã được thực hiện.
* Các lỗi nghiêm trọng đã được xử lý.
* Pipeline chạy End-to-End thành công.
* Có kết quả Detection và Segmentation.
* Có metrics đánh giá.
* Có hình ảnh trực quan để đưa vào báo cáo.
