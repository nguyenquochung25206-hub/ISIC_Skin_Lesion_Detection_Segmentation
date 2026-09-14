# Test Cases

## 1. Mục đích

Kiểm thử hệ thống **ISIC Skin Lesion Detection & Segmentation** nhằm đảm bảo các chức năng chính hoạt động đúng:

```text
Input Image
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
```

---

## 2. Test Case

| ID   | Chức năng                  | Input                           | Kết quả mong đợi                    | Trạng thái |
| ---- | -------------------------- | ------------------------------- | ----------------------------------- | ---------- |
| TC01 | Đọc ảnh                    | Ảnh ISIC hợp lệ                 | Ảnh được đọc thành công             | ☐          |
| TC02 | Đọc ảnh lỗi                | File không phải ảnh             | Hệ thống báo lỗi, không crash       | ☐          |
| TC03 | Resize                     | Ảnh kích thước bất kỳ           | Ảnh được đưa về kích thước quy định | ☐          |
| TC04 | Normalize                  | Ảnh RGB                         | Pixel được chuẩn hóa đúng           | ☐          |
| TC05 | Faster R-CNN               | Ảnh có lesion                   | Phát hiện được Bounding Box         | ☐          |
| TC06 | Detection Confidence       | Ảnh có lesion                   | Confidence được xuất ra             | ☐          |
| TC07 | Detection không có kết quả | Ảnh không phát hiện được lesion | Hệ thống xử lý an toàn              | ☐          |
| TC08 | Crop ROI                   | Bounding Box hợp lệ             | Cắt đúng vùng lesion                | ☐          |
| TC09 | U-Net                      | ROI lesion                      | Sinh được segmentation mask         | ☐          |
| TC10 | Mask Output                | Prediction mask                 | Mask có kích thước đúng             | ☐          |
| TC11 | Threshold                  | Probability mask                | Chuyển thành binary mask 0/1        | ☐          |
| TC12 | Dice                       | Ground Truth + Prediction       | Tính được Dice                      | ☐          |
| TC13 | IoU                        | Ground Truth + Prediction       | Tính được IoU                       | ☐          |
| TC14 | Precision                  | Ground Truth + Prediction       | Tính được Precision                 | ☐          |
| TC15 | Recall                     | Ground Truth + Prediction       | Tính được Recall                    | ☐          |
| TC16 | Visualization              | Input + Prediction              | Hiển thị được kết quả trực quan     | ☐          |
| TC17 | Overlay                    | Ảnh + Mask                      | Tạo được ảnh overlay                | ☐          |
| TC18 | Batch Testing              | Nhiều ảnh test                  | Hệ thống xử lý lần lượt không lỗi   | ☐          |
| TC19 | Lưu kết quả                | Ảnh/mask/metrics                | Kết quả được lưu đúng thư mục       | ☐          |
| TC20 | End-to-End                 | Ảnh ISIC                        | Hoàn thành toàn bộ pipeline         | ☐          |

---

## 3. Kiểm thử Detection

### TC05 – Faster R-CNN Detection

**Input:** Ảnh dermoscopic có tổn thương da.

**Các bước:**

1. Đưa ảnh vào hệ thống.
2. Thực hiện preprocessing.
3. Chạy Faster R-CNN.
4. Lấy Bounding Box và Confidence.

**Kết quả mong đợi:**

```text
Bounding Box = [xmin, ymin, xmax, ymax]
Confidence > threshold
```

Hệ thống xác định được vị trí tương đối của lesion.

---

## 4. Kiểm thử Segmentation

### TC09 – U-Net Segmentation

**Input:** ROI được cắt từ Bounding Box.

**Các bước:**

1. Đưa ROI vào U-Net.
2. Model dự đoán probability mask.
3. Áp dụng threshold.
4. Tạo binary segmentation mask.

**Kết quả mong đợi:**

```text
Output shape = H × W × 1
Background = 0
Lesion = 1
```

---

## 5. Kiểm thử Evaluation

Với mỗi ảnh test, hệ thống so sánh:

```text
Ground Truth Mask
        ↓
     Evaluation
        ↑
Predicted Mask
```

Các metric cần kiểm tra:

| Metric    | Mục đích                                  |
| --------- | ----------------------------------------- |
| Dice      | Đo mức độ chồng lấp giữa hai mask         |
| IoU       | Đo tỷ lệ giao / hợp                       |
| Precision | Đánh giá mức dự đoán đúng vùng lesion     |
| Recall    | Đánh giá khả năng phát hiện đầy đủ lesion |

---

## 6. Kiểm thử Visualization

Kết quả nên hiển thị tối thiểu:

```text
┌──────────────┬──────────────┐
│ Original     │ Ground Truth │
├──────────────┼──────────────┤
│ Prediction   │ Overlay      │
└──────────────┴──────────────┘
```

Mục đích là kiểm tra trực quan xem segmentation có bám đúng vùng tổn thương hay không.

---

## 7. Kiểm thử End-to-End

### TC20 – Full Pipeline

**Input:** Một ảnh ISIC chưa xử lý.

**Pipeline:**

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
Post-processing
    ↓
Evaluation
    ↓
Save Results
```

**Kết quả mong đợi:**

Hệ thống hoàn thành toàn bộ pipeline mà không xảy ra lỗi và tạo được:

```text
results/
├── predictions/
├── visualizations/
├── metrics/
└── plots/
```

---

## 8. Tiêu chí Pass / Fail

### PASS

Test được xem là **PASS** khi:

* Hệ thống chạy đúng chức năng.
* Không xảy ra lỗi ngoài dự kiến.
* Output có đúng định dạng.
* Kết quả segmentation có thể đánh giá bằng các metric.
* Kết quả được lưu đúng vị trí.

### FAIL

Test được xem là **FAIL** khi:

* Chương trình bị crash.
* Không đọc hoặc xử lý được input hợp lệ.
* Không tạo được Bounding Box khi model có detection.
* Không tạo được segmentation mask.
* Kết quả có kích thước hoặc định dạng sai.
* Không lưu được kết quả.

---

## 9. Test Report

Sau khi kiểm thử, nhóm tổng hợp kết quả:

| Tổng số Test | PASS | FAIL | Tỷ lệ PASS |
| -----------: | ---: | ---: | ---------: |
|           20 |      |      |            |

**Người kiểm thử:** Thành viên phụ trách Testing
**Ngày kiểm thử:** `YYYY-MM-DD`
**Version:** `v1.0`
