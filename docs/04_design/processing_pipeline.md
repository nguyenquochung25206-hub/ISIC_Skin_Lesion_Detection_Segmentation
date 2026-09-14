# Processing Pipeline

## 1. Tổng quan

Processing Pipeline mô tả quy trình xử lý ảnh của hệ thống phát hiện và phân vùng tổn thương da dựa trên bộ dữ liệu ISIC.

Toàn bộ quá trình được thực hiện theo chuỗi:

```text
Input Image
    ↓
Preprocessing
    ↓
Detection - Faster R-CNN
    ↓
Detection Result
    ↓
Segmentation - U-Net
    ↓
Predicted Mask
    ↓
Evaluation
    ↓
Visualization
    ↓
Final Result
```

Pipeline giúp kết nối các module trong hệ thống thành một quy trình xử lý thống nhất.

---

## 2. Input

Hệ thống nhận ảnh da liễu từ bộ dữ liệu ISIC.

Dữ liệu đầu vào gồm:

* Ảnh da liễu.
* Ground Truth Mask tương ứng với ảnh.

Cấu trúc dữ liệu:

```text
data/
└── input/
    ├── images/
    └── masks/
```

Ảnh trong `images/` được sử dụng làm dữ liệu đầu vào cho quá trình xử lý.

Các mask trong `masks/` được sử dụng làm Ground Truth để đánh giá kết quả Segmentation.

---

## 3. Preprocessing

Sau khi nhận ảnh đầu vào, hệ thống thực hiện bước tiền xử lý.

Mục đích của bước này là đưa dữ liệu về định dạng phù hợp trước khi đưa vào các mô hình.

Các thao tác có thể bao gồm:

1. Đọc ảnh từ thư mục dữ liệu.
2. Kiểm tra dữ liệu đầu vào.
3. Resize ảnh.
4. Chuẩn hóa giá trị pixel.
5. Chuẩn bị Ground Truth Mask.
6. Thực hiện Data Augmentation trong trường hợp cần tăng tính đa dạng của dữ liệu.

Module preprocessing được đặt tại:

```text
src/preprocessing/
```

Sau preprocessing, dữ liệu được lưu hoặc sử dụng tại:

```text
data/processed/
```

---

## 4. Detection bằng Faster R-CNN

Sau preprocessing, ảnh được đưa vào Detection Module.

Mô hình được sử dụng là **Faster R-CNN**.

Mục tiêu của bước Detection là xác định vị trí vùng tổn thương trên ảnh.

Kết quả Detection gồm:

* Bounding Box.
* Confidence Score.
* Thông tin về vùng được phát hiện.

Có thể hình dung:

```text
Ảnh gốc
   ↓
Faster R-CNN
   ↓
Bounding Box
   +
Confidence Score
```

Bounding Box giúp xác định vị trí tương đối của vùng tổn thương.

---

## 5. Segmentation bằng U-Net

Sau khi có kết quả Detection, hệ thống thực hiện bước Segmentation.

Mô hình được sử dụng là **U-Net**.

Mục tiêu của Segmentation là xác định chính xác các pixel thuộc vùng tổn thương.

Khác với Bounding Box của Detection, Segmentation tạo ra một mask biểu diễn vùng tổn thương theo từng pixel.

Quy trình:

```text
Detection Result
       ↓
      U-Net
       ↓
Predicted Mask
```

Predicted Mask là kết quả phân vùng do mô hình dự đoán.

---

## 6. Evaluation

Sau khi U-Net tạo Predicted Mask, hệ thống so sánh kết quả này với Ground Truth Mask.

Các chỉ số chính được sử dụng cho Segmentation là:

* IoU.
* Dice Score.

### 6.1. IoU

IoU đo mức độ chồng lấp giữa Predicted Mask và Ground Truth Mask.

```text
              Prediction ∩ Ground Truth
IoU = -------------------------------------------
              Prediction ∪ Ground Truth
```

IoU càng cao thì hai vùng càng giống nhau.

Giá trị IoU nằm trong khoảng:

```text
0 ≤ IoU ≤ 1
```

---

### 6.2. Dice Score

Dice Score cũng được sử dụng để đo mức độ tương đồng giữa hai mask.

```text
                 2 × Intersection
Dice = -------------------------------------
             Prediction Area + GT Area
```

Giá trị Dice nằm trong khoảng:

```text
0 ≤ Dice ≤ 1
```

Giá trị càng gần 1 thể hiện kết quả phân vùng càng tốt.

---

## 7. Visualization

Sau khi hoàn thành Evaluation, hệ thống trực quan hóa kết quả.

Các thông tin có thể được hiển thị gồm:

* Ảnh gốc.
* Detection Bounding Box.
* Ground Truth Mask.
* Predicted Mask.
* Kết quả so sánh Ground Truth và Prediction.
* Các chỉ số IoU và Dice.

Ví dụ quy trình:

```text
Original Image
      ↓
Ground Truth Mask
      ↓
Predicted Mask
      ↓
Comparison
      ↓
Visualization
```

Visualization giúp người sử dụng dễ dàng quan sát chất lượng của kết quả thay vì chỉ dựa vào các giá trị số.

---

## 8. Saving Results

Các kết quả sau quá trình xử lý được lưu vào các thư mục tương ứng.

Kết quả visualization:

```text
data/output/visualizations/
```

Các hình ảnh phục vụ báo cáo:

```text
results/figures/segmentation/
```

Các chỉ số đánh giá:

```text
results/tables/segmentation_metrics.csv
```

Việc lưu kết quả giúp nhóm có thể sử dụng lại dữ liệu trong quá trình phân tích và viết báo cáo.

---

## 9. Complete Processing Flow

Toàn bộ pipeline của hệ thống có thể được biểu diễn như sau:

```text
+------------------+
|   ISIC Dataset   |
+--------+---------+
         |
         v
+------------------+
|  Preprocessing   |
| Resize/Normalize|
+--------+---------+
         |
         v
+------------------+
|   Faster R-CNN   |
|    Detection     |
+--------+---------+
         |
         v
+------------------+
| Detection Result |
| BBox + Confidence|
+--------+---------+
         |
         v
+------------------+
|      U-Net       |
|   Segmentation   |
+--------+---------+
         |
         v
+------------------+
|  Predicted Mask  |
+--------+---------+
         |
         v
+------------------+
|    Evaluation    |
|   IoU / Dice     |
+--------+---------+
         |
         v
+------------------+
|  Visualization   |
+--------+---------+
         |
         v
+------------------+
|   Final Results  |
+------------------+
```

---

## 10. Pipeline Execution

Khi chương trình được thực thi, dữ liệu được xử lý theo các bước:

### Step 1 - Load Data

Hệ thống đọc ảnh và Ground Truth Mask từ bộ dữ liệu.

### Step 2 - Preprocess

Ảnh được resize, chuẩn hóa và chuẩn bị cho mô hình.

### Step 3 - Detection

Faster R-CNN phát hiện vùng tổn thương và tạo Bounding Box cùng Confidence Score.

### Step 4 - Segmentation

U-Net nhận dữ liệu đầu vào và tạo Predicted Mask.

### Step 5 - Evaluation

Predicted Mask được so sánh với Ground Truth Mask để tính IoU và Dice Score.

### Step 6 - Visualization

Các kết quả được hiển thị dưới dạng hình ảnh để dễ quan sát.

### Step 7 - Save Results

Kết quả hình ảnh và các chỉ số được lưu vào thư mục kết quả của project.

---

## 11. Kết quả đầu ra của Pipeline

Sau khi pipeline hoàn thành, hệ thống có thể tạo ra:

```text
Input
  ↓
Processed Image
  ↓
Detection Result
  ├── Bounding Box
  └── Confidence Score
  ↓
Segmentation Result
  └── Predicted Mask
  ↓
Evaluation
  ├── IoU
  └── Dice Score
  ↓
Visualization
  └── Result Images
```

Các kết quả này được sử dụng để đánh giá khả năng phát hiện và phân vùng tổn thương da của hệ thống.

---

## 12. Mối liên hệ giữa các module

Pipeline được thiết kế theo hướng module hóa.

Mỗi module nhận dữ liệu từ bước trước và cung cấp kết quả cho bước tiếp theo:

```text
Preprocessing
      ↓
Detection
      ↓
Segmentation
      ↓
Evaluation
      ↓
Visualization
```

Cách tổ chức này giúp các thành viên trong nhóm có thể phát triển từng module độc lập và sau đó tích hợp thành hệ thống hoàn chỉnh.

Đồng thời, việc tách riêng các module giúp quá trình kiểm thử, sửa lỗi và mở rộng hệ thống dễ dàng hơn.

