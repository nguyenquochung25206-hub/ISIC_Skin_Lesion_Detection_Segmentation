# U-Net

## 1. Tổng quan

**U-Net** là kiến trúc Deep Learning được thiết kế cho bài toán **Image Segmentation**, đặc biệt phù hợp với ảnh y tế.

Trong đề tài **ISIC Skin Lesion Detection and Segmentation**, U-Net được sử dụng để phân vùng chính xác vùng tổn thương da trên ảnh dermoscopic.

Mục tiêu:

```text
Ảnh đầu vào
    ↓
U-Net
    ↓
Segmentation Mask
```

---

## 2. Kiến trúc U-Net

U-Net gồm hai phần chính:

```text
Encoder → Decoder
```

Có thể mô tả:

```text
Input Image
     │
     ▼
  Encoder
     │
     ▼
Bottleneck
     │
     ▼
  Decoder
     │
     ▼
Output Mask
```

### Encoder

Encoder giảm kích thước ảnh và trích xuất các đặc trưng quan trọng như:

* Biên.
* Hình dạng.
* Texture.
* Đặc điểm của vùng tổn thương.

### Bottleneck

Bottleneck là phần sâu nhất của mạng, chứa các đặc trưng cấp cao của ảnh.

### Decoder

Decoder khôi phục kích thước không gian của ảnh để tạo ra segmentation mask.

---

## 3. Skip Connection

Đặc điểm quan trọng của U-Net là **Skip Connection**.

Skip Connection truyền đặc trưng từ Encoder trực tiếp sang Decoder:

```text
Encoder Feature
      │
      ├──────────────┐
      │              │
      ▼              ▼
Bottleneck       Decoder
                     │
                     ▼
                  Output
```

Điều này giúp Decoder giữ được thông tin vị trí và chi tiết của vùng tổn thương.

---

## 4. Input và Output

### Input

Ảnh dermoscopic RGB:

```text
H × W × 3
```

Ví dụ:

```text
256 × 256 × 3
```

### Output

U-Net tạo ra một segmentation mask:

```text
H × W × 1
```

Mỗi pixel thể hiện xác suất thuộc vùng lesion.

Sau khi threshold:

```text
Probability < 0.5 → Background (0)

Probability ≥ 0.5 → Skin Lesion (1)
```

---

## 5. Training

Quá trình huấn luyện:

```text
Image
  ↓
Preprocessing
  ↓
U-Net
  ↓
Predicted Mask
  ↓
So sánh với Ground Truth
  ↓
Calculate Loss
  ↓
Backpropagation
  ↓
Update Model
```

Quá trình được lặp lại qua nhiều epoch cho đến khi mô hình đạt kết quả ổn định.

---

## 6. Loss Function

Các loss function có thể sử dụng:

```text
BCE Loss
Dice Loss
BCE + Dice Loss
```

Trong project, **BCE + Dice Loss** có thể được sử dụng để kết hợp khả năng phân loại pixel của BCE và khả năng tối ưu vùng chồng lấp của Dice.

---

## 7. Data Augmentation

Để tăng khả năng tổng quát hóa, có thể áp dụng augmentation cho training data:

* Horizontal Flip.
* Vertical Flip.
* Rotation.
* Random Crop.
* Brightness/Contrast.

Ảnh và mask phải được biến đổi đồng thời để đảm bảo chúng vẫn tương ứng.

---

## 8. Evaluation

Kết quả U-Net được đánh giá chủ yếu bằng:

```text
Dice Coefficient
IoU
Precision
Recall
```

Trong đó:

**Dice** và **IoU** là hai chỉ số quan trọng nhất đối với segmentation.

```text
Ground Truth Mask
        │
        │ Compare
        ▼
Predicted Mask
        │
        ▼
Dice / IoU / Precision / Recall
```

---

## 9. Visualization

Để đánh giá trực quan, kết quả nên được hiển thị:

```text
┌──────────────────┬──────────────────┐
│ Original Image   │ Ground Truth     │
├──────────────────┼──────────────────┤
│ Predicted Mask   │ Overlay          │
└──────────────────┴──────────────────┘
```

Qua visualization có thể quan sát:

* Mô hình phân vùng đúng.
* Phần lesion bị bỏ sót.
* Phần background bị dự đoán nhầm.
* Độ chính xác của đường biên lesion.

---

## 10. Vai trò trong hệ thống

U-Net đảm nhiệm nhiệm vụ **Segmentation** trong project.

Faster R-CNN:

```text
Xác định lesion nằm ở đâu
→ Bounding Box
```

U-Net:

```text
Xác định chính xác pixel nào thuộc lesion
→ Segmentation Mask
```

Pipeline tổng thể:

```text
ISIC Image
    ↓
Preprocessing
    ↓
Faster R-CNN
    ↓
Lesion Bounding Box
    ↓
U-Net
    ↓
Lesion Segmentation Mask
    ↓
Dice / IoU / Precision / Recall
```

## 11. Kết quả mong đợi

U-Net cần tạo ra segmentation mask có mức độ tương đồng cao với Ground Truth.

Mô hình tốt cần:

* Phân vùng đúng vị trí lesion.
* Hạn chế bỏ sót vùng lesion.
* Hạn chế dự đoán dư background.
* Giữ đường biên lesion tương đối chính xác.
* Đạt Dice và IoU cao trên tập Test.
