# Faster R-CNN

## 1. Tổng quan

**Faster R-CNN (Faster Region-based Convolutional Neural Network)** là mô hình Deep Learning dùng cho bài toán **Object Detection**. Trong đề tài ISIC Skin Lesion Detection and Segmentation, Faster R-CNN được sử dụng để xác định vị trí vùng tổn thương da thông qua **Bounding Box**.

```text
Ảnh đầu vào
    ↓
Backbone CNN
    ↓
Feature Map
    ↓
Region Proposal Network (RPN)
    ↓
Region Proposals
    ↓
RoI Align
    ↓
Detection Head
    ↓
Bounding Box + Class + Confidence
```

## 2. Các thành phần chính

### Backbone

Backbone CNN có nhiệm vụ trích xuất các đặc trưng từ ảnh đầu vào như hình dạng, đường biên, màu sắc và texture.

### Region Proposal Network (RPN)

RPN tạo ra các vùng đề xuất có khả năng chứa tổn thương dựa trên Feature Map.

### RoI Align

RoI Align trích xuất đặc trưng từ các Region Proposal và đưa chúng về kích thước phù hợp để phân loại.

### Detection Head

Detection Head thực hiện hai nhiệm vụ:

* **Classification:** xác định vùng có phải Skin Lesion hay không.
* **Bounding Box Regression:** điều chỉnh tọa độ Bounding Box.

## 3. Input và Output

**Input:**

```text
Ảnh dermoscopic RGB
```

**Output:**

```text
Bounding Box
Class
Confidence Score
```

Ví dụ:

```text
Box: [x1, y1, x2, y2]
Class: Skin Lesion
Confidence: 0.95
```

## 4. Annotation

Faster R-CNN yêu cầu dữ liệu huấn luyện có **Bounding Box Ground Truth**.

Nếu dataset ISIC chỉ cung cấp Segmentation Mask, Bounding Box có thể được tạo từ mask bằng cách tìm tọa độ:

```text
xmin = min(x)
ymin = min(y)
xmax = max(x)
ymax = max(y)
```

Bounding Box thu được sẽ bao quanh vùng tổn thương.

## 5. Huấn luyện

Trong quá trình training, mô hình học đồng thời:

```text
RPN Classification
RPN Bounding Box Regression
Detection Classification
Detection Bounding Box Regression
```

Từ đó mô hình học cách phát hiện và định vị vùng tổn thương.

## 6. Đánh giá

Các chỉ số chính sử dụng cho Detection:

* **IoU:** đo mức độ chồng lấp giữa Bounding Box dự đoán và Ground Truth.
* **Precision:** tỷ lệ dự đoán đúng trong tổng số dự đoán.
* **Recall:** khả năng phát hiện các tổn thương thực tế.
* **mAP:** đánh giá tổng thể chất lượng Object Detection.

Ví dụ với IoU threshold:

```text
IoU ≥ 0.5 → Detection được xem là đúng.
```

## 7. Vai trò trong hệ thống

Faster R-CNN đảm nhiệm bước **Detection**, trong khi mô hình Segmentation đảm nhiệm việc xác định chính xác từng pixel của vùng tổn thương.

```text
Input Image
     ↓
Faster R-CNN
     ↓
Lesion Bounding Box
     ↓
Segmentation Model
     ↓
Lesion Mask
```

Faster R-CNN giúp xác định **tổn thương nằm ở đâu**, còn Segmentation xác định **chính xác vùng pixel nào thuộc tổn thương**.
