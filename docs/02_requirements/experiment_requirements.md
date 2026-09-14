# Experiment Requirements

## 1. Mục đích

Tài liệu này mô tả các yêu cầu cần thiết để thực hiện các thí nghiệm trong đề tài **ISIC Skin Lesion Detection and Segmentation**.

Mục tiêu của thí nghiệm là xây dựng và đánh giá hệ thống xử lý ảnh y tế có khả năng xác định vùng tổn thương da trong ảnh dermoscopic và phân vùng chính xác khu vực tổn thương.

Các thí nghiệm tập trung vào việc đánh giá chất lượng của quá trình tiền xử lý ảnh, mô hình phân vùng và khả năng tổng quát hóa của mô hình trên tập dữ liệu ISIC.

---

## 2. Dataset

### 2.1. Dataset sử dụng

Thí nghiệm sử dụng dữ liệu ảnh tổn thương da từ **ISIC (International Skin Imaging Collaboration)**.

Mỗi mẫu dữ liệu gồm:

* Ảnh dermoscopic của vùng da.
* Ground Truth Mask tương ứng đối với bài toán segmentation.
* Thông tin nhãn hoặc metadata nếu dataset cung cấp.

Ảnh được sử dụng làm đầu vào cho hệ thống, trong khi Ground Truth Mask được sử dụng để huấn luyện và đánh giá kết quả segmentation.

### 2.2. Cấu trúc dữ liệu đề xuất

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

Trong đó:

```text
images/train/ → ảnh dùng để huấn luyện
images/val/   → ảnh dùng để validation
images/test/  → ảnh dùng để kiểm tra cuối cùng

masks/train/  → Ground Truth Mask của train
masks/val/    → Ground Truth Mask của validation
masks/test/   → Ground Truth Mask của test
```

Ảnh và mask phải có quan hệ tương ứng với nhau.

Ví dụ:

```text
images/train/ISIC_0000001.jpg
masks/train/ISIC_0000001.png
```

---

## 3. Yêu cầu phần cứng

Thí nghiệm có thể được thực hiện trên máy tính cá nhân.

### 3.1. Cấu hình tối thiểu

* CPU: Intel Core i5 hoặc tương đương.
* RAM: 8 GB trở lên.
* GPU: Không bắt buộc nếu chỉ thực hiện thử nghiệm nhỏ.
* Storage: tối thiểu 10 GB trống.

### 3.2. Cấu hình khuyến nghị

* CPU: Intel Core i5/i7 hoặc AMD Ryzen 5/7.
* RAM: 16 GB trở lên.
* GPU NVIDIA hỗ trợ CUDA.
* VRAM: từ 6 GB trở lên.
* Storage: SSD, tối thiểu 20 GB trống.

GPU được khuyến nghị để giảm thời gian huấn luyện mô hình segmentation.

---

## 4. Yêu cầu phần mềm

### 4.1. Hệ điều hành

Có thể sử dụng:

* Windows 10/11
* Ubuntu Linux
* Các hệ điều hành tương thích với Python và PyTorch.

### 4.2. Ngôn ngữ lập trình

Sử dụng:

```text
Python 3.10+
```

Khuyến nghị sử dụng môi trường virtual environment để tránh xung đột thư viện.

Ví dụ:

```bash
python -m venv .venv
```

Kích hoạt môi trường trên Windows:

```bash
.venv\Scripts\activate
```

---

## 5. Thư viện yêu cầu

Các thư viện chính dự kiến sử dụng:

```text
numpy
pandas
opencv-python
Pillow
matplotlib
scikit-learn
torch
torchvision
tqdm
```

Nếu sử dụng thêm các mô hình hoặc framework hỗ trợ segmentation, có thể bổ sung:

```text
segmentation-models-pytorch
albumentations
```

Phiên bản cụ thể của các thư viện cần được ghi lại trong:

```text
requirements.txt
```

Mục đích là đảm bảo các thành viên trong nhóm có thể cài đặt cùng một môi trường và tái lập kết quả thí nghiệm.

---

## 6. Tiền xử lý dữ liệu

Trước khi đưa ảnh vào mô hình, dữ liệu cần được tiền xử lý để đảm bảo tính đồng nhất.

Các bước tiền xử lý dự kiến gồm:

### 6.1. Resize

Đưa tất cả ảnh về cùng kích thước.

Ví dụ:

```text
256 × 256
```

hoặc:

```text
512 × 512
```

Kích thước cuối cùng phụ thuộc vào cấu hình mô hình và khả năng của GPU.

### 6.2. Normalization

Chuẩn hóa giá trị pixel nhằm giúp mô hình hội tụ ổn định hơn trong quá trình huấn luyện.

### 6.3. Data Augmentation

Có thể áp dụng các phép biến đổi để tăng tính đa dạng của dữ liệu huấn luyện:

* Horizontal Flip.
* Vertical Flip.
* Rotation.
* Random Crop.
* Scale.
* Brightness/Contrast adjustment.

Data augmentation chỉ được áp dụng cho tập training.

Không áp dụng augmentation ngẫu nhiên cho test set để đảm bảo kết quả đánh giá khách quan.

---

## 7. Chia tập dữ liệu

Dataset được chia thành ba tập:

```text
Training Set
Validation Set
Test Set
```

Tỷ lệ đề xuất:

```text
Training   : 70%
Validation : 15%
Testing    : 15%
```

hoặc sử dụng đúng cách chia tập được quy định bởi dataset/challenge nếu dataset cung cấp sẵn train/validation/test.

### Training Set

Được sử dụng để huấn luyện mô hình.

### Validation Set

Được sử dụng để:

* Theo dõi quá trình huấn luyện.
* Điều chỉnh hyperparameter.
* Phát hiện overfitting.
* Lựa chọn model checkpoint tốt nhất.

### Test Set

Chỉ sử dụng sau khi hoàn thành quá trình huấn luyện và lựa chọn mô hình.

Kết quả trên test set được sử dụng để đánh giá cuối cùng.

---

## 8. Mô hình thực nghiệm

### 8.1. Bài toán Segmentation

Mô hình segmentation được sử dụng để dự đoán vùng tổn thương trên ảnh.

Mô hình cơ sở đề xuất:

```text
U-Net
```

U-Net phù hợp với bài toán segmentation ảnh y tế vì kiến trúc có encoder-decoder và sử dụng skip connection để giữ lại thông tin không gian của ảnh.

Input:

```text
Ảnh dermoscopic
```

Output:

```text
Binary Segmentation Mask
```

Trong đó:

```text
0 → Background
1 → Skin Lesion
```

---

## 9. Loss Function

Trong quá trình huấn luyện segmentation, có thể sử dụng:

```text
Binary Cross Entropy Loss
```

hoặc:

```text
Dice Loss
```

Một lựa chọn phù hợp là kết hợp:

```text
BCE Loss + Dice Loss
```

Mục đích của việc kết hợp hai loss là tận dụng khả năng tối ưu pixel của BCE và khả năng xử lý sự mất cân bằng giữa foreground/background của Dice Loss.

Loss được tính trên tập training và validation để theo dõi quá trình học.

---

## 10. Hyperparameters

Các hyperparameter cần được ghi nhận trong mỗi thí nghiệm:

| Parameter      | Giá trị đề xuất |
| -------------- | --------------: |
| Image Size     |       256 × 256 |
| Batch Size     |       8 hoặc 16 |
| Epochs         |          30–100 |
| Learning Rate  |           0.001 |
| Optimizer      |            Adam |
| Loss           |      BCE + Dice |
| Random Seed    |         Cố định |
| Early Stopping |  Có thể sử dụng |

Các giá trị trên có thể được thay đổi trong quá trình thực nghiệm.

Mọi thay đổi cần được ghi lại để có thể so sánh giữa các thí nghiệm.

---

## 11. Evaluation Metrics

Kết quả segmentation được đánh giá bằng các chỉ số sau.

### 11.1. Dice Coefficient

Dice đo mức độ tương đồng giữa Ground Truth Mask và Predicted Mask.

Công thức:

```text
Dice = 2 × |Prediction ∩ Ground Truth|
       --------------------------------
       |Prediction| + |Ground Truth|
```

Dice càng gần 1 thì kết quả segmentation càng tốt.

---

### 11.2. Intersection over Union

IoU đo tỷ lệ giữa phần giao và phần hợp của prediction và ground truth.

```text
IoU = |Prediction ∩ Ground Truth|
      ----------------------------
      |Prediction ∪ Ground Truth|
```

IoU càng cao thì mô hình càng phân vùng chính xác.

---

### 11.3. Precision

Precision cho biết trong các pixel được mô hình dự đoán là lesion, có bao nhiêu pixel thực sự thuộc lesion.

```text
Precision = TP / (TP + FP)
```

---

### 11.4. Recall

Recall cho biết trong tổng số pixel lesion thực tế, mô hình phát hiện được bao nhiêu pixel.

```text
Recall = TP / (TP + FN)
```

---

## 12. Kết quả cần lưu

Mỗi thí nghiệm cần lưu lại các kết quả quan trọng.

### 12.1. Model

```text
models/
└── best_model.pth
```

Model checkpoint tốt nhất cần được lưu để có thể sử dụng lại mà không cần huấn luyện từ đầu.

### 12.2. Training History

Cần lưu:

```text
train_loss
val_loss
train_dice
val_dice
```

Có thể lưu dưới dạng:

```text
CSV
JSON
```

hoặc sử dụng để vẽ biểu đồ.

### 12.3. Prediction

Lưu một số kết quả dự đoán đại diện:

```text
results/
├── predictions/
├── masks/
└── visualizations/
```

---

## 13. Visualization

Để đánh giá trực quan, cần tạo hình ảnh so sánh:

```text
Original Image
       │
       ▼
Ground Truth Mask
       │
       ▼
Predicted Mask
       │
       ▼
Overlay
```

Ví dụ bố cục:

```text
+----------------+----------------+
| Original Image | Ground Truth   |
+----------------+----------------+
| Prediction     | Overlay        |
+----------------+----------------+
```

Visualization giúp nhóm dễ dàng nhận biết các trường hợp:

* Phân vùng chính xác.
* Phân vùng thiếu.
* Phân vùng dư.
* Sai vùng tổn thương.
* Boundary không chính xác.

---

## 14. Các thí nghiệm cần thực hiện

Nhóm nên thực hiện ít nhất các thí nghiệm sau.

### Experiment 01 – Baseline

Huấn luyện mô hình với preprocessing cơ bản.

Mục tiêu:

```text
Xác định kết quả baseline của mô hình.
```

### Experiment 02 – Data Augmentation

Thêm các phép augmentation vào tập training.

Mục tiêu:

```text
Đánh giá ảnh hưởng của data augmentation.
```

### Experiment 03 – Image Size

So sánh các kích thước ảnh khác nhau.

Ví dụ:

```text
256 × 256
512 × 512
```

Mục tiêu:

```text
Đánh giá ảnh hưởng của độ phân giải đầu vào.
```

### Experiment 04 – Loss Function

So sánh:

```text
BCE Loss
Dice Loss
BCE + Dice Loss
```

Mục tiêu:

```text
Xác định loss function phù hợp với bài toán.
```

### Experiment 05 – Final Model

Huấn luyện mô hình với cấu hình tốt nhất được lựa chọn từ các thí nghiệm trước.

Kết quả của experiment này được sử dụng làm kết quả chính trong báo cáo.

---

## 15. Bảng ghi nhận kết quả

Mỗi thí nghiệm cần được ghi lại theo bảng:

| Experiment | Model | Image Size | Loss       | Augmentation | Dice | IoU | Precision | Recall |
| ---------- | ----- | ---------- | ---------- | ------------ | ---: | --: | --------: | -----: |
| EXP01      | U-Net | 256×256    | BCE        | No           |    - |   - |         - |      - |
| EXP02      | U-Net | 256×256    | BCE        | Yes          |    - |   - |         - |      - |
| EXP03      | U-Net | 512×512    | BCE        | Yes          |    - |   - |         - |      - |
| EXP04      | U-Net | 256×256    | BCE + Dice | Yes          |    - |   - |         - |      - |
| EXP05      | Final | Best       | Best       | Yes          |    - |   - |         - |      - |

Các giá trị `-` sẽ được thay thế bằng kết quả thực tế sau khi chạy thí nghiệm.

---

## 16. Reproducibility

Để đảm bảo kết quả có thể tái lập, cần lưu:

* Python version.
* Library versions.
* Dataset version.
* Dataset split.
* Image size.
* Batch size.
* Learning rate.
* Number of epochs.
* Optimizer.
* Loss function.
* Random seed.
* Model architecture.
* Hardware sử dụng.

Random seed cần được cố định khi có thể để giảm sự khác biệt giữa các lần chạy.

---

## 17. Tiêu chí hoàn thành thí nghiệm

Thí nghiệm được xem là hoàn thành khi:

1. Dataset được tải và kiểm tra thành công.
2. Ảnh và Ground Truth Mask được ghép đúng.
3. Pipeline preprocessing hoạt động.
4. Mô hình có thể training thành công.
5. Validation được thực hiện trong quá trình training.
6. Model checkpoint tốt nhất được lưu.
7. Test set được đánh giá.
8. Dice và IoU được tính toán.
9. Prediction được trực quan hóa.
10. Kết quả của các experiment được ghi lại.
11. Có thể tái chạy thí nghiệm từ môi trường đã khai báo.
12. Kết quả cuối cùng được sử dụng trong báo cáo và thuyết trình.

---

## 18. Expected Output

Sau khi hoàn thành các thí nghiệm, project cần có các output chính:

```text
results/
├── metrics/
│   └── experiment_results.csv
│
├── predictions/
│   ├── sample_01.png
│   ├── sample_02.png
│   └── ...
│
├── visualizations/
│   ├── comparison_01.png
│   ├── comparison_02.png
│   └── ...
│
└── plots/
    ├── loss_curve.png
    └── dice_curve.png
```

Kết quả cuối cùng cần cho phép nhóm trả lời được các câu hỏi:

```text
1. Mô hình có phân vùng được tổn thương da hay không?

2. Mô hình đạt Dice bao nhiêu?

3. Mô hình đạt IoU bao nhiêu?

4. Data augmentation có cải thiện kết quả không?

5. Loss function nào cho kết quả tốt nhất?

6. Kích thước ảnh nào phù hợp nhất?

7. Những trường hợp nào mô hình dự đoán sai?

8. Mô hình có thể được sử dụng để hỗ trợ phân vùng tổn thương
   trên ảnh da hay không?
```

---

## 19. Lưu ý

Đây là bài toán xử lý ảnh/thị giác máy tính trên dữ liệu y tế. Kết quả của mô hình chỉ được sử dụng cho mục đích học tập và nghiên cứu trong phạm vi đồ án.

Mô hình không được xem là công cụ chẩn đoán y khoa và kết quả dự đoán không thay thế đánh giá của bác sĩ hoặc chuyên gia y tế.
