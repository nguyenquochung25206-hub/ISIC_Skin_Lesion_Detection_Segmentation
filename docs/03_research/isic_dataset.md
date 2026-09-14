# ISIC Dataset

## 1. Tổng quan

ISIC Dataset là bộ dữ liệu ảnh da liễu được sử dụng để nghiên cứu và phát triển các hệ thống phân tích tổn thương da bằng Computer Vision.

Trong project, dataset được sử dụng cho hai nhiệm vụ:

- Detection: phát hiện vị trí tổn thương bằng Faster R-CNN.
- Segmentation: phân đoạn vùng tổn thương bằng U-Net.

## 2. Thành phần dữ liệu

Dataset gồm:

```text
data/
├── images/
│   ├── train/
│   └── test/
│
└── masks/
    ├── train/
    └── test/

images/: ảnh dermoscopic đầu vào.

masks/: mask biểu diễn vùng tổn thương da, dùng làm Ground Truth cho segmentation.

3. Bounding Box

Faster R-CNN cần bounding box. Trong project, bounding box được tạo từ segmentation mask.

Segmentation Mask
        ↓
Tìm vùng lesion
        ↓
[x_min, y_min, x_max, y_max]
        ↓
Ground Truth Bounding Box

Class sử dụng:

0 = Background
1 = Skin Lesion
4. Preprocessing

Ảnh được xử lý trước khi đưa vào mô hình:

Ảnh gốc
  ↓
Convert RGB
  ↓
Resize 256 × 256
  ↓
Normalize pixel 0–255 → 0–1
5. Data Augmentation

Augmentation chỉ áp dụng cho tập training nhằm tăng tính đa dạng của dữ liệu.

Các phép biến đổi:

Horizontal Flip
Vertical Flip
Rotation
Brightness
Contrast

Đối với các phép biến đổi hình học, image và mask phải được biến đổi giống nhau để giữ đúng vị trí tổn thương.

6. Sử dụng cho Detection

Faster R-CNN sử dụng:

Image + Bounding Box + Class Label

Kết quả:

Bounding Box
Confidence Score
Class
7. Sử dụng cho Segmentation

U-Net sử dụng:

Image → U-Net → Predicted Mask

Mask Ground Truth được dùng để tính các chỉ số đánh giá.

8. Đánh giá

Detection sử dụng:

IoU
Precision
Recall
F1-score

Segmentation sử dụng:

Dice
IoU
Precision
Recall
F1-score
Accuracy
9. Lưu ý
Không sử dụng dữ liệu test để huấn luyện.
Image và mask phải tương ứng với nhau.
Không áp dụng augmentation ngẫu nhiên cho test set.
Khi resize mask nên sử dụng Nearest Neighbor để giữ nguyên giá trị nhãn.
10. Pipeline
ISIC Dataset
     ↓
Preprocessing
     ↓
Data Augmentation
     ↓
┌───────────────┬───────────────┐
↓               ↓
Faster R-CNN    U-Net
↓               ↓
Detection       Segmentation
└───────────────┴───────────────┘
             ↓
         Evaluation
             ↓
        Final Results
