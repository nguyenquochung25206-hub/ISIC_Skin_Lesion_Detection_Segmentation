# Conclusion

## 1. Kết luận

Đề tài **ISIC Skin Lesion Detection & Segmentation** xây dựng một hệ thống kết hợp **Faster R-CNN** và **U-Net** để phát hiện và phân vùng tổn thương da từ ảnh dermoscopic.

Trong đó, Faster R-CNN thực hiện phát hiện vị trí tổn thương thông qua **Bounding Box**, còn U-Net thực hiện phân vùng ở mức pixel để tạo **Segmentation Mask**. Hai giai đoạn được kết hợp thành một pipeline hoàn chỉnh từ ảnh đầu vào đến kết quả cuối cùng.

```text
ISIC Image
    ↓
Preprocessing
    ↓
Faster R-CNN
    ↓
Bounding Box
    ↓
ROI Crop
    ↓
U-Net
    ↓
Segmentation Mask
    ↓
Evaluation
```

Kết quả được đánh giá thông qua các chỉ số **Dice, IoU, Precision và Recall**, đồng thời sử dụng hình ảnh trực quan để so sánh giữa Ground Truth và kết quả dự đoán.

---

## 2. Kết quả đạt được

Dự án đã xây dựng được các thành phần chính:

* Xử lý và chuẩn bị dữ liệu ISIC.
* Tiền xử lý ảnh và chuẩn hóa dữ liệu.
* Xây dựng quy trình Detection bằng Faster R-CNN.
* Xác định Bounding Box của tổn thương.
* Cắt vùng ROI để đưa vào bước Segmentation.
* Xây dựng quy trình Segmentation bằng U-Net.
* Tạo và xử lý Segmentation Mask.
* Tính toán các Evaluation Metrics.
* Trực quan hóa kết quả bằng Prediction và Overlay.
* Xây dựng quy trình kiểm thử cho toàn bộ hệ thống.

---

## 3. Hạn chế

Hệ thống vẫn còn một số hạn chế:

* Hiệu quả phụ thuộc vào chất lượng và số lượng dữ liệu huấn luyện.
* Kết quả Detection có thể ảnh hưởng trực tiếp đến chất lượng Segmentation nếu Bounding Box không chính xác.
* U-Net có thể gặp khó khăn với các lesion có hình dạng phức tạp hoặc ranh giới không rõ.
* Thời gian huấn luyện phụ thuộc nhiều vào GPU và cấu hình phần cứng.
* Hệ thống hiện chủ yếu tập trung vào Detection và Segmentation, chưa thực hiện phân loại chính xác loại bệnh lý.

---

## 4. Hướng phát triển

Trong tương lai, hệ thống có thể được cải thiện theo các hướng:

* Sử dụng dataset lớn hơn và đa dạng hơn.
* Tối ưu preprocessing và data augmentation.
* Fine-tune các backbone mạnh hơn cho Faster R-CNN.
* Cải thiện U-Net bằng Attention U-Net hoặc các kiến trúc segmentation hiện đại.
* Thử nghiệm các phương pháp kết hợp Detection và Segmentation khác.
* Tối ưu tốc độ suy luận.
* Xây dựng giao diện cho phép người dùng tải ảnh và xem kết quả.
* Bổ sung bước **classification** để hỗ trợ xác định loại tổn thương da.

---

## 5. Tổng kết

Đề tài cho thấy việc kết hợp **Object Detection** và **Image Segmentation** có thể tạo thành một quy trình xử lý ảnh y tế hoàn chỉnh. Faster R-CNN giúp xác định **tổn thương nằm ở đâu**, trong khi U-Net xác định **những pixel nào thuộc tổn thương**.

Qua quá trình thực hiện, nhóm có cơ hội áp dụng các kiến thức về **Computer Vision, Deep Learning, Image Processing và Model Evaluation** vào một bài toán thực tế trên dữ liệu ISIC.

Hệ thống là nền tảng để tiếp tục phát triển các giải pháp hỗ trợ phân tích ảnh da bằng AI trong các nghiên cứu tiếp theo.
