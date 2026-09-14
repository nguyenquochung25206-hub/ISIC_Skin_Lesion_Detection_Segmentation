import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn


# ============================================================
# FASTER R-CNN MODEL
# ============================================================

def create_faster_rcnn(
    num_classes=2,
    pretrained=False
):
    """
    Tao mo hinh Faster R-CNN voi ResNet-50 FPN.

    Parameters
    ----------
    num_classes : int
        So luong class bao gom background.
        2 = background + skin lesion.

    pretrained : bool
        Su dung pretrained COCO hay khong.

    Returns
    -------
    model : torch.nn.Module
        Mo hinh Faster R-CNN.
    """

    if pretrained:

        model = fasterrcnn_resnet50_fpn(
            weights="DEFAULT"
        )

        # Thay prediction head de phu hop voi dataset ISIC
        in_features = (
            model.roi_heads.box_predictor.cls_score
            .in_features
        )

        from torchvision.models.detection.faster_rcnn import (
            FastRCNNPredictor
        )

        model.roi_heads.box_predictor = (
            FastRCNNPredictor(
                in_features,
                num_classes
            )
        )

    else:

        model = fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes
        )

    return model


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_faster_rcnn(
    model_path,
    num_classes=2,
    device=None
):
    """
    Load Faster R-CNN tu file checkpoint.

    Parameters
    ----------
    model_path : str or Path
        Duong dan den file .pth.

    num_classes : int
        So class cua model.

    device : torch.device
        CPU hoac CUDA.

    Returns
    -------
    model
        Faster R-CNN da load.
    """

    if device is None:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    model = create_faster_rcnn(
        num_classes=num_classes,
        pretrained=False
    )

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    # --------------------------------------------------------
    # Truong hop checkpoint la dictionary
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.to(device)

    model.eval()

    return model


# ============================================================
# PREDICT
# ============================================================

def predict(
    model,
    image,
    device=None,
    score_threshold=0.5
):
    """
    Du doan bounding box tren mot anh.

    Parameters
    ----------
    model : Faster R-CNN model

    image : Tensor
        Anh dang torch Tensor [C, H, W].

    device : torch.device
        CPU hoac CUDA.

    score_threshold : float
        Nguong confidence.

    Returns
    -------
    prediction : dict
        Gom boxes, labels, scores.
    """

    if device is None:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    model.eval()

    image = image.to(device)

    with torch.no_grad():

        output = model(
            [image]
        )[0]

    scores = output["scores"]

    keep = (
        scores
        >= score_threshold
    )

    prediction = {
        "boxes": output["boxes"][keep],
        "labels": output["labels"][keep],
        "scores": output["scores"][keep]
    }

    return prediction


# ============================================================
# GET BEST PREDICTION
# ============================================================

def get_best_prediction(
    prediction
):
    """
    Lay bounding box co confidence cao nhat.

    Returns
    -------
    bbox : list or None
    score : float
    """

    boxes = prediction["boxes"]

    scores = prediction["scores"]

    if len(boxes) == 0:

        return None, 0.0

    best_index = torch.argmax(
        scores
    )

    bbox = (
        boxes[best_index]
        .detach()
        .cpu()
        .numpy()
        .tolist()
    )

    score = float(
        scores[best_index]
        .detach()
        .cpu()
        .item()
    )

    return bbox, score


# ============================================================
# MODEL INFORMATION
# ============================================================

def print_model_info(
    model
):
    """
    In thong tin co ban cua model.
    """

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("=" * 60)
    print("FASTER R-CNN MODEL")
    print("=" * 60)

    print(
        "Total parameters:",
        f"{total_parameters:,}"
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}"
    )

    print(
        "Training mode:",
        model.training
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    model = create_faster_rcnn(
        num_classes=2,
        pretrained=False
    )

    model.to(device)

    print_model_info(
        model
    )
