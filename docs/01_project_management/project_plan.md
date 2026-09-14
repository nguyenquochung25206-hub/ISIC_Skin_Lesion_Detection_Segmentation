# Project Plan – ISIC Skin Lesion Detection & Segmentation

## 1. Tổng quan

Dự án xây dựng hệ thống xử lý ảnh y tế nhằm phát hiện và phân đoạn
tổn thương da từ ảnh da liễu thuộc bộ dữ liệu ISIC.

Project tập trung vào hai nhiệm vụ chính:

- Detection: phát hiện vị trí tổn thương da bằng Faster R-CNN.
- Segmentation: phân đoạn vùng tổn thương da bằng U-Net.

## 2. Mục tiêu

### Mục tiêu tổng quát

Xây dựng pipeline hoàn chỉnh từ dữ liệu đầu vào, tiền xử lý,
huấn luyện mô hình, dự đoán, đánh giá và trực quan hóa kết quả.

### Mục tiêu cụ thể

- Khảo sát bộ dữ liệu ISIC.
- Tiền xử lý và chuẩn hóa dữ liệu.
- Xây dựng mô hình Faster R-CNN cho Detection.
- Xây dựng mô hình U-Net cho Segmentation.
- Đánh giá Detection bằng Precision, Recall, mAP.
- Đánh giá Segmentation bằng IoU, Dice.
- Trực quan hóa kết quả.
- Kiểm thử toàn bộ pipeline.
- Tổng hợp và phân tích kết quả.

## 3. Phạm vi

Project bao gồm:

1. Dataset ISIC.
2. Image preprocessing.
3. Data augmentation.
4. Object detection.
5. Image segmentation.
6. Evaluation.
7. Visualization.
8. Testing.
9. Báo cáo kết quả.

## 4. Công nghệ

- Python
- PyTorch
- torchvision
- OpenCV
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Jupyter Notebook
- Git/GitHub

## 5. Mô hình

### Detection

Sử dụng Faster R-CNN để xác định bounding box
của vùng tổn thương da.

### Segmentation

Sử dụng U-Net để tạo segmentation mask,
xác định chính xác vùng tổn thương.

## 6. Quy trình thực hiện

Dataset
→ Preprocessing
→ Detection / Segmentation
→ Prediction
→ Evaluation
→ Visualization
→ Final Results

## 7. Kết quả đầu ra

Project tạo ra:

- Bounding boxes.
- Segmentation masks.
- Evaluation metrics.
- Hình ảnh trực quan hóa.
- Bảng kết quả.
- Báo cáo tổng hợp.

## 8. Quản lý project

Project được quản lý bằng Git/GitHub.

Mỗi thành viên chịu trách nhiệm một module chính,
sau đó tích hợp các module thành một pipeline hoàn chỉnh.
