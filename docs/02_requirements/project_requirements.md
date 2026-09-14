# Yêu cầu chính của đề tài

## 1. Tên đề tài
Phát hiện và phân đoạn tổn thương da trên ảnh y tế sử dụng Deep Learning (ISIC Dataset)

## 2. Đặt vấn đề
Ung thư da (đặc biệt là u hắc tố melanoma) là một trong những loại ung thư nguy hiểm nếu không được phát hiện sớm. Việc tự động hóa quá trình phát hiện vị trí và phân đoạn chính xác vùng tổn thương da từ ảnh chụp da liễu giúp hỗ trợ bác sĩ chẩn đoán nhanh và chính xác hơn.

## 3. Mục tiêu đề tài
- Xây dựng mô hình **Detection** (Faster R-CNN) để xác định vị trí (bounding box) của vùng tổn thương da trong ảnh.
- Xây dựng mô hình **Segmentation** (U-Net) để phân đoạn chính xác đường viền (mask) của vùng tổn thương.
- Đánh giá và so sánh hiệu năng hai mô hình bằng các chỉ số chuẩn trong lĩnh vực computer vision y tế.

## 4. Yêu cầu chức năng
1. Đọc và tiền xử lý ảnh/mask từ bộ dữ liệu ISIC (resize, normalize, augmentation).
2. Huấn luyện mô hình Faster R-CNN cho bài toán detection.
3. Huấn luyện mô hình U-Net cho bài toán segmentation.
4. Dự đoán bounding box và mask trên ảnh mới (tập validation/test).
5. Tính toán các chỉ số đánh giá:
   - Detection: Precision, Recall, mAP (mean Average Precision).
   - Segmentation: IoU (Intersection over Union), Dice coefficient.
6. Trực quan hóa kết quả (vẽ bounding box, overlay mask lên ảnh gốc).
7. Lưu trữ kết quả (hình ảnh, bảng số liệu) phục vụ báo cáo.

## 5. Yêu cầu phi chức năng
- Code có cấu trúc module rõ ràng, có thể tái sử dụng (`src/` tách theo preprocessing, detection, segmentation, evaluation, visualization).
- Có unit test cho các thành phần chính (`tests/`).
- Có tài liệu đầy đủ: nghiên cứu lý thuyết, thiết kế hệ thống, kế hoạch/kết quả kiểm thử, báo cáo kết quả.
- Kết quả có thể tái lập (reproducible): cố định seed, ghi rõ cấu hình huấn luyện.

## 6. Dữ liệu sử dụng
- Nguồn: [ISIC Archive](https://www.isic-archive.com/)
- Bao gồm: ảnh tổn thương da (dermoscopic images) và ground-truth segmentation mask tương ứng.
- Chi tiết tìm hiểu tại [`../03_research/isic_dataset.md`](../03_research/isic_dataset.md).

## 7. Tiêu chí đánh giá thành công
- Mô hình Faster R-CNN đạt mAP ở mức chấp nhận được trên tập test (mục tiêu cụ thể sẽ xác định sau khi có baseline).
- Mô hình U-Net đạt chỉ số Dice/IoU ở mức chấp nhận được trên tập test.
- Có báo cáo, bảng số liệu và hình ảnh trực quan minh họa đầy đủ cho cả hai mô hình.

## 8. Yêu cầu bàn giao (deliverables)
- Source code đầy đủ trong `src/`, `experiments/`, `tests/`.
- Tài liệu trong `docs/` (nghiên cứu, thiết kế, testing, kết quả).
- Báo cáo tổng hợp cuối cùng: [`../../results/reports/final_results.md`](../../results/reports/final_results.md).

## 9. Tài liệu liên quan
- [`experiment_requirements.md`](experiment_requirements.md) — Yêu cầu chi tiết về thí nghiệm và đánh giá (TV7)
- [`../01_project_management/project_plan.md`](../01_project_management/project_plan.md) — Kế hoạch thực hiện
