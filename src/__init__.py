"""
ISIC Skin Lesion Detection & Segmentation
==========================================

Package chính của project, bao gồm các module:
- preprocessing: tiền xử lý ảnh và data augmentation
- detection: xây dựng, huấn luyện, dự đoán bằng Faster R-CNN
- segmentation: xây dựng, huấn luyện, dự đoán bằng U-Net
- evaluation: tính toán các chỉ số đánh giá (Precision, Recall, mAP, IoU, Dice)
- visualization: hiển thị và lưu kết quả (bounding box, mask)
"""

__version__ = "0.1.0"
__author__ = "ISIC Skin Lesion Detection & Segmentation Team"
