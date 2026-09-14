# Segmentation Results

## 1. Mục đích

Tài liệu ghi nhận kết quả **Image Segmentation** của mô hình **U-Net** trên dữ liệu tổn thương da ISIC.

Mục tiêu là xác định chính xác vùng lesion ở mức **pixel** và tạo ra **Segmentation Mask**.

---

## 2. Model

| Thành phần | Thiết lập                |
| ---------- | ------------------------ |
| Model      | U-Net                    |
| Task       | Image Segmentation       |
| Input      | ROI / ảnh dermoscopic    |
| Output     | Binary Segmentation Mask |
| Object     | Skin Lesion              |
| Dataset    | ISIC                     |

Pipeline:

```text id="h9c9gv"
Input / ROI
    ↓
Preprocessing
    ↓
U-Net
    ↓
Probability Mask
    ↓
Threshold
    ↓
Binary Segmentation Mask
```

---

## 3. Segmentation Output

Mỗi ảnh tạo ra một predicted mask:

```text id="b7i2oe"
Probability Mask
       ↓
Threshold = 0.5
       ↓
Binary Mask

Background = 0
Lesion     = 1
```

Ví dụ:

```text id="y8w3rk"
Image: ISIC_XXXXXXX.jpg
Output: Segmentation Mask
Threshold: 0.5
```

---

## 4. Evaluation Metrics

Các chỉ số sử dụng:

| Metric    | Ý nghĩa                                             |
| --------- | --------------------------------------------------- |
| Dice      | Đo mức độ chồng lấp giữa Prediction và Ground Truth |
| IoU       | Đo tỷ lệ giao nhau trên hợp                         |
| Precision | Độ chính xác của vùng lesion được dự đoán           |
| Recall    | Khả năng phát hiện đầy đủ vùng lesion               |

---

## 5. Overall Results

| Metric    | Value |
| --------- | ----: |
| Dice      |  `--` |
| IoU       |  `--` |
| Precision |  `--` |
| Recall    |  `--` |

> Các giá trị được cập nhật sau khi chạy trên tập test.

---

## 6. Sample Results

| Image    | Ground Truth | Prediction | Dice | IoU |
| -------- | ------------ | ---------- | ---: | --: |
| ISIC_001 | Mask         | Mask       |   -- |  -- |
| ISIC_002 | Mask         | Mask       |   -- |  -- |
| ISIC_003 | Mask         | Mask       |   -- |  -- |

---

## 7. Visualization

Mỗi kết quả nên được hiển thị theo dạng:

```text id="3dq7mc"
┌──────────────┬──────────────┐
│ Original     │ Ground Truth │
├──────────────┼──────────────┤
│ Prediction   │ Overlay      │
└──────────────┴──────────────┘
```

Trong đó:

* **Original:** Ảnh ISIC ban đầu.
* **Ground Truth:** Mask chuẩn từ dataset.
* **Prediction:** Mask do U-Net dự đoán.
* **Overlay:** Mask được chồng lên ảnh gốc để quan sát trực quan.

---

## 8. Result Directory

Kết quả Segmentation được lưu tại:

```text id="t1qz6g"
results/
└── segmentation/
    ├── predictions/
    ├── visualizations/
    └── metrics/
```

---

## 9. Error Analysis

Các trường hợp cần kiểm tra:

| Trường hợp         | Mô tả                                |
| ------------------ | ------------------------------------ |
| Good Segmentation  | Prediction gần giống Ground Truth    |
| Over-segmentation  | Dự đoán vùng lesion lớn hơn thực tế  |
| Under-segmentation | Bỏ sót một phần lesion               |
| Boundary Error     | Biên lesion dự đoán không chính xác  |
| False Positive     | Dự đoán nhầm background thành lesion |
| False Negative     | Bỏ sót vùng lesion                   |

Các trường hợp có Dice hoặc IoU thấp cần được xem xét để tìm nguyên nhân.

---

## 10. Comparison

Có thể tổng hợp kết quả theo từng ảnh:

| Image    |   Dice |    IoU | Precision | Recall |
| -------- | -----: | -----: | --------: | -----: |
| ISIC_001 |     -- |     -- |        -- |     -- |
| ISIC_002 |     -- |     -- |        -- |     -- |
| ISIC_003 |     -- |     -- |        -- |     -- |
| **Mean** | **--** | **--** |    **--** | **--** |

---

## 11. Kết luận

U-Net thực hiện nhiệm vụ **phân vùng tổn thương da ở mức pixel**.

Kết quả Segmentation được đánh giá bằng **Dice, IoU, Precision và Recall**, đồng thời kiểm tra trực quan thông qua Ground Truth, Prediction và Overlay.

Kết quả từ bước Detection và Segmentation được kết hợp để đánh giá toàn bộ hệ thống:

```text id="7n7d0v"
Faster R-CNN
     ↓
Bounding Box
     ↓
ROI
     ↓
U-Net
     ↓
Segmentation Mask
     ↓
Dice / IoU / Precision / Recall
```
