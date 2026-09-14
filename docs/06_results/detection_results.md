# Detection Results

## 1. Mục đích

Tài liệu ghi nhận kết quả **Object Detection** của mô hình **Faster R-CNN** trên tập dữ liệu ISIC.

Mục tiêu là xác định vị trí tổn thương da thông qua **Bounding Box** và đánh giá độ chính xác của model.

---

## 2. Model

| Thành phần | Thiết lập                 |
| ---------- | ------------------------- |
| Model      | Faster R-CNN              |
| Task       | Object Detection          |
| Input      | Ảnh dermoscopic RGB       |
| Output     | Bounding Box + Confidence |
| Object     | Skin Lesion               |
| Dataset    | ISIC                      |

Pipeline:

```text
Input Image
    ↓
Preprocessing
    ↓
Faster R-CNN
    ↓
Bounding Box
    ↓
Confidence Score
```

---

## 3. Detection Output

Mỗi ảnh được dự đoán với thông tin:

```text
Bounding Box = [xmin, ymin, xmax, ymax]
Class = Lesion
Confidence = 0.xx
```

Ví dụ:

```text
Image: ISIC_XXXXXXX.jpg
Class: Lesion
Confidence: 0.95
Bounding Box: [52, 38, 421, 398]
```

---

## 4. Evaluation Metrics

Các chỉ số sử dụng để đánh giá Detection:

| Metric    | Ý nghĩa                                                    |
| --------- | ---------------------------------------------------------- |
| IoU       | Mức độ chồng lấp giữa Bounding Box dự đoán và Ground Truth |
| Precision | Tỷ lệ detection đúng trong các detection được dự đoán      |
| Recall    | Khả năng phát hiện đúng các lesion                         |
| mAP       | Đánh giá tổng thể hiệu năng Object Detection               |

---

## 5. Overall Results

| Metric       | Value |
| ------------ | ----: |
| IoU          |  `--` |
| Precision    |  `--` |
| Recall       |  `--` |
| mAP@0.5      |  `--` |
| mAP@0.5:0.95 |  `--` |

> Các giá trị được cập nhật sau khi hoàn thành quá trình testing.

---

## 6. Sample Results

| Image    | Ground Truth | Prediction   | Confidence |
| -------- | ------------ | ------------ | ---------: |
| ISIC_001 | Bounding Box | Bounding Box |         -- |
| ISIC_002 | Bounding Box | Bounding Box |         -- |
| ISIC_003 | Bounding Box | Bounding Box |         -- |

Kết quả trực quan nên thể hiện:

```text
Original Image
      ↓
Predicted Bounding Box
      ↓
Ground Truth vs Prediction
```

---

## 7. Detection Visualization

Kết quả cần lưu tại:

```text
results/
└── detection/
    ├── predictions/
    ├── visualizations/
    └── metrics/
```

Ảnh visualization nên hiển thị:

```text
┌─────────────────────────────┐
│        ISIC Image           │
│                             │
│    ┌─────────────────┐      │
│    │     Lesion      │      │
│    │                 │      │
│    └─────────────────┘      │
│       Confidence: 0.95      │
└─────────────────────────────┘
```

---

## 8. Error Analysis

Các trường hợp cần kiểm tra:

| Trường hợp        | Mô tả                                       |
| ----------------- | ------------------------------------------- |
| True Positive     | Phát hiện đúng lesion                       |
| False Positive    | Phát hiện nhầm vùng không phải lesion       |
| False Negative    | Bỏ sót lesion                               |
| Poor Localization | Bounding Box không bao phủ chính xác lesion |

Các lỗi thường gặp cần được ghi nhận để cải thiện model.

---

## 9. Kết luận

Faster R-CNN được sử dụng để xác định **vị trí của tổn thương da** trước khi chuyển vùng ROI sang U-Net.

Kết quả Detection sẽ được sử dụng làm đầu vào cho bước **Segmentation**:

```text
Faster R-CNN
     ↓
Bounding Box
     ↓
Crop ROI
     ↓
U-Net
```

Hiệu năng cuối cùng được đánh giá dựa trên **IoU, Precision, Recall và mAP**.
