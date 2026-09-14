"""
[TV6] Hiển thị / trực quan hóa kết quả phân đoạn.

Cung cấp:
    - overlay_mask      : tô màu vùng mask lên ảnh gốc
    - draw_contour      : vẽ viền contour của mask
    - make_side_by_side : ghép ảnh gốc | mask | overlay
    - visualize_segmentation : hàm tiện lợi hiển thị bằng matplotlib
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------
def overlay_mask(image_rgb, mask, color=(255, 0, 0), alpha=0.45):
    """
    Chồng mask lên ảnh gốc.
    Args:
        image_rgb: np.ndarray (H,W,3) uint8, RGB
        mask     : np.ndarray (H,W) uint8, giá trị 0 hoặc 255
        color    : (R,G,B) màu tô vùng mask
        alpha    : độ trong suốt (0-1)
    Returns:
        overlay: np.ndarray (H,W,3) uint8 RGB
    """
    if mask.shape[:2] != image_rgb.shape[:2]:
        mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    bin_mask = (mask > 127).astype(np.uint8)

    # Tạo lớp màu chỉ tại vùng mask
    colored = np.zeros_like(image_rgb, dtype=np.uint8)
    colored[bin_mask == 1] = color

    # Kết hợp có trọng số, chỉ ở vùng mask (giữ nguyên nền)
    overlay = image_rgb.copy()
    idx = bin_mask == 1
    overlay[idx] = (image_rgb[idx].astype(np.float32) * (1 - alpha)
                    + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
    return overlay


# ---------------------------------------------------------------------------
# Contour
# ---------------------------------------------------------------------------
def draw_contour(image_rgb, mask, color=(0, 255, 0), thickness=2):
    """
    Vẽ viền contour của mask lên ảnh (RGB).
    """
    if mask.shape[:2] != image_rgb.shape[:2]:
        mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    bin_mask = (mask > 127).astype(np.uint8)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    out = image_rgb.copy()
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


# ---------------------------------------------------------------------------
# Side-by-side
# ---------------------------------------------------------------------------
def make_side_by_side(image_rgb, mask, overlay=None,
                      titles=("Original", "Predicted Mask", "Overlay")):
    """
    Ghép 3 ảnh cạnh nhau (theo chiều ngang) và gắn title.
    Trả về np.ndarray (H, 3W, 3) uint8 RGB.
    """
    if overlay is None:
        overlay = overlay_mask(image_rgb, mask)

    # Mask -> RGB để ghép
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    h = image_rgb.shape[0]
    # Resize mask và overlay cho bằng chiều cao ảnh gốc
    def _fit(img):
        if img.shape[0] != h:
            scale = h / img.shape[0]
            img = cv2.resize(img, (int(img.shape[1] * scale), h))
        return img

    imgs = [_fit(image_rgb), _fit(mask_rgb), _fit(overlay)]
    canvas = np.concatenate(imgs, axis=1)

    # Ghi title
    x = 10
    for i, t in enumerate(titles):
        cv2.putText(canvas, t, (x + i * imgs[0].shape[1] + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    return canvas


# ---------------------------------------------------------------------------
# Visualize bằng matplotlib (tiện cho notebook)
# ---------------------------------------------------------------------------
def visualize_segmentation(image_rgb, mask, save_path=None, show=True,
                           color=(255, 0, 0), alpha=0.45):
    """
    Hiển thị 3 panel: ảnh gốc | mask | overlay + contour.
    """
    overlay = overlay_mask(image_rgb, mask, color=color, alpha=alpha)
    overlay = draw_contour(overlay, mask, color=(0, 255, 0), thickness=2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Ảnh gốc")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Mask dự đoán")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (đỏ) + Contour (xanh)")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Visualize] Đã lưu hình -> {save_path}")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    # Demo với ảnh giả
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(mask, (128, 128), 60, 255, -1)
    visualize_segmentation(img, mask, show=False, save_path="/tmp/demo.png")
