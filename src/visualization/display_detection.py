
"""
[TV4] Trực quan hóa kết quả Detection (Faster R-CNN).

Cung cấp:
    - draw_boxes            : vẽ nhiều bounding box lên ảnh
    - draw_single_box       : vẽ 1 box
    - make_detection_panel  : ghép ảnh gốc | ảnh có box
    - visualize_detection   : hiển thị bằng matplotlib
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ===========================================================================
# Bảng màu cố định cho các class (BGR)
# ===========================================================================
COLORS = {
    "MEL":   (0, 0, 255),      # Đỏ   - Melanoma
    "NV":    (0, 200, 0),      # Xanh lá - Nevus
    "BCC":   (255, 0, 0),      # Xanh dương - Basal cell carcinoma
    "AK":    (0, 165, 255),    # Cam - Actinic keratosis
    "BKL":   (255, 0, 255),    # Tím - Benign keratosis
    "DF":    (0, 255, 255),    # Vàng - Dermatofibroma
    "VASC":  (255, 255, 0),    # Cyan - Vascular lesion
    "SCC":   (128, 0, 128),    # Tím đậm - Squamous cell carcinoma
    "UNK":   (128, 128, 128),  # Xám - Unknown
}


def get_color(class_name, fallback=(0, 255, 0)):
    """Trả về màu BGR cho class, có fallback nếu không có trong bảng."""
    return COLORS.get(class_name.upper(), fallback)


# ===========================================================================
# Vẽ bounding box
# ===========================================================================
def draw_single_box(image, box, label="", score=None, color=(0, 255, 0),
                    thickness=2, font_scale=0.6, show_score=True):
    """
    Vẽ 1 bounding box lên ảnh (BGR, in-place qua bản copy).

    Args:
        image     : np.ndarray (H,W,3) BGR
        box       : (x1, y1, x2, y2)
        label     : tên class
        score     : độ tin cậy (0-1)
        color     : màu BGR
        thickness : độ dày nét
        font_scale: kích thước chữ
        show_score: có hiển thị score hay không
    Returns:
        ảnh đã vẽ (BGR)
    """
    out = image.copy()
    x1, y1, x2, y2 = map(int, box)

    # Vẽ hình chữ nhật
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

    # Tạo text: "MEL 0.95"
    text = label
    if show_score and score is not None:
        text = f"{label} {score:.2f}" if label else f"{score:.2f}"

    if text:
        # Đo kích thước text để vẽ nền
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness - 1)
        )
        # Nền đen phía sau chữ (giúp chữ nổi bật)
        cv2.rectangle(out, (x1, y1 - th - baseline - 4),
                      (x1 + tw + 6, y1), color, -1)
        # Chữ trắng
        cv2.putText(out, text, (x1 + 3, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), max(1, thickness - 1),
                    lineType=cv2.LINE_AA)
    return out


def draw_boxes(image, boxes, labels=None, scores=None,
               class_names=None, thickness=2, font_scale=0.6):
    """
    Vẽ nhiều bounding box lên ảnh.

    Args:
        image       : np.ndarray (H,W,3) BGR
        boxes       : list/array (N, 4) — (x1, y1, x2, y2)
        labels      : list[int] (N,) — class id (tùy chọn)
        scores      : list[float] (N,) — confidence (tùy chọn)
        class_names : list[str] — map id → tên class (tùy chọn)
                      VD: ["MEL", "NV", "BCC", ...]
        thickness   : độ dày nét
        font_scale  : kích thước chữ
    Returns:
        ảnh đã vẽ (BGR)
    """
    out = image.copy()

    if boxes is None or len(boxes) == 0:
        return out

    boxes = np.asarray(boxes).reshape(-1, 4)
    n = len(boxes)

    for i in range(n):
        # Xác định label
        if labels is not None and len(labels) > i:
            cls_id = int(labels[i])
            if class_names and 0 <= cls_id < len(class_names):
                label = class_names[cls_id]
            else:
                label = str(cls_id)
        else:
            label = ""

        # Xác định score
        score = float(scores[i]) if (scores is not None and len(scores) > i) else None

        # Màu theo class
        color = get_color(label, fallback=(0, 255, 0))

        out = draw_single_box(
            out, boxes[i], label=label, score=score,
            color=color, thickness=thickness, font_scale=font_scale,
        )
    return out


# ===========================================================================
# Ghép panel: ảnh gốc | ảnh có box
# ===========================================================================
def make_detection_panel(image_bgr, boxes=None, labels=None, scores=None,
                         class_names=None, titles=("Original", "Detection")):
    """
    Ghép 2 ảnh cạnh nhau: gốc | có bounding box.
    Nếu truyền thêm mask (tùy chọn) sẽ ghép thêm 1 cột.
    """
    img_with_box = draw_boxes(image_bgr, boxes, labels, scores,
                              class_names=class_names)

    # Resize 2 ảnh bằng nhau về chiều cao
    h = image_bgr.shape[0]
    if img_with_box.shape[0] != h:
        scale = h / img_with_box.shape[0]
        img_with_box = cv2.resize(img_with_box,
                                  (int(img_with_box.shape[1] * scale), h))

    canvas = np.concatenate([image_bgr, img_with_box], axis=1)

    # Vẽ title
    cv2.putText(canvas, titles[0], (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(canvas, titles[1],
                (image_bgr.shape[1] + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return canvas


# ===========================================================================
# Hiển thị bằng matplotlib (tiện cho notebook)
# ===========================================================================
def visualize_detection(image_bgr, boxes=None, labels=None, scores=None,
                        class_names=None, save_path=None, show=True):
    """
    Hiển thị 2 panel: Ảnh gốc | Ảnh có bounding box.
    """
    # Chuyển BGR -> RGB để hiển thị đúng với matplotlib
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_with_box = draw_boxes(image_bgr, boxes, labels, scores,
                              class_names=class_names)
    img_box_rgb = cv2.cvtColor(img_with_box, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Ảnh gốc")
    axes[0].axis("off")

    axes[1].imshow(img_box_rgb)
    axes[1].set_title("Detection (bounding box)")
    axes[1].axis("off")

    # Legend cho các class có mặt
    if labels is not None and class_names is not None:
        unique_ids = sorted(set(int(l) for l in labels))
        patches = [
            mpatches.Patch(
                color=tuple(c / 255.0 for c in get_color(class_names[i])[::-1]),
                label=class_names[i]
            )
            for i in unique_ids if 0 <= i < len(class_names)
        ]
        if patches:
            axes[1].legend(handles=patches, loc="lower right",
                           fontsize=9, framealpha=0.8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Display-Detection] Đã lưu hình -> {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ===========================================================================
# Tiện ích: lọc box theo ngưỡng confidence
# ===========================================================================
def filter_by_score(boxes, scores, labels=None, threshold=0.5):
    """Lọc bớt box có score thấp hơn ngưỡng."""
    if scores is None:
        return boxes, labels, scores
    scores = np.asarray(scores)
    keep = scores >= threshold
    boxes = np.asarray(boxes)[keep]
    labels_out = np.asarray(labels)[keep] if labels is not None else None
    return boxes, labels_out, scores[keep]


# ===========================================================================
# Demo
# ===========================================================================
if __name__ == "__main__":
    # Tạo ảnh giả
    img = np.random.randint(0, 200, (400, 600, 3), dtype=np.uint8)

    # Box giả
    boxes = np.array([
        [50, 60, 200, 220],
        [300, 100, 480, 300],
    ])
    labels = [0, 1]                    # MEL, NV
    scores = [0.95, 0.78]
    class_names = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC"]

    visualize_detection(
        img, boxes, labels, scores,
        class_names=class_names,
        save_path="demo_detection.png",
        show=False,
    )
    print("✅ Demo hoàn tất -> demo_detection.png")
