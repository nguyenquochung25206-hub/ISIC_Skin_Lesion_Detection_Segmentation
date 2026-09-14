import os
import cv2


def save_image(image, output_path):
    """
    Lưu một hình ảnh vào đường dẫn được chỉ định.

    Parameters:
        image: Hình ảnh cần lưu.
        output_path: Đường dẫn và tên file đầu ra.

    Returns:
        True nếu lưu thành công, False nếu lưu thất bại.
    """

    # Tạo thư mục chứa file nếu thư mục chưa tồn tại
    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Lưu hình ảnh
    success = cv2.imwrite(output_path, image)

    return success


def save_segmentation_result(
    original_image,
    ground_truth_mask,
    predicted_mask,
    output_dir,
    image_name
):
    """
    Lưu các kết quả segmentation.

    Parameters:
        original_image: Ảnh gốc.
        ground_truth_mask: Ground Truth Mask.
        predicted_mask: Predicted Mask.
        output_dir: Thư mục lưu kết quả.
        image_name: Tên ảnh.

    Returns:
        Dictionary chứa đường dẫn của các file đã lưu.
    """

    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)

    # Tạo tên file cho từng kết quả
    original_path = os.path.join(
        output_dir,
        f"{image_name}_original.png"
    )

    ground_truth_path = os.path.join(
        output_dir,
        f"{image_name}_ground_truth.png"
    )

    predicted_path = os.path.join(
        output_dir,
        f"{image_name}_prediction.png"
    )

    # Lưu từng hình ảnh
    save_image(original_image, original_path)
    save_image(ground_truth_mask, ground_truth_path)
    save_image(predicted_mask, predicted_path)

    # Trả về đường dẫn các file
    return {
        "original": original_path,
        "ground_truth": ground_truth_path,
        "prediction": predicted_path
    }
