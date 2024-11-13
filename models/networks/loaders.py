from arrow import get
import torch
import torchvision
from typing import Callable
import importlib
MODEL_LIST = [
    "deeplabv3_baseline",
    "deeplabv3_sp",
    "deeplabv3_mrb",
    "deeplabv3_all",
    "maskrcnn",
    "unet",
    "res_unet++",
    "attention_unet",
]
class ModelSelector:
    def __init__(self, num_classes: int, pretrained: bool = True):
        self.num_classes = num_classes
        self.pretrained = pretrained

    def select_model(self, model_name: str, checkpoint_path: str = None)-> tuple[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]: 
        if model_name not in MODEL_LIST:
            raise ValueError(f"Model {model_name} not found in the list of available models")
        if self.pretrained and checkpoint_path is None:
            raise ValueError("Pretrained model requested but no checkpoint path provided")
        if "deeplabv3" in model_name:
            return self._load_deeplab_models(model_name, checkpoint_path), lambda x: x
        elif "maskrcnn" in model_name:
            return self._load_maskrcnn(model_name, checkpoint_path), maskrcnn_prediction
        elif "unet" in model_name:
            return self._load_unet(model_name, checkpoint_path), lambda x: x
        pass 

    def _load_deeplab_models(self, model_name: str, checkpoint_path: str):
        if "baseline" in model_name:
            from segmentation_models_pytorch import DeepLabV3Plus
            if checkpoint_path is None:
                return DeepLabV3Plus(
                    encoder_name='resnet101', 
                    encoder_weights=None, 
                    classes=4, 
                    activation='softmax2d',
                )
            else:
                return torch.load(checkpoint_path)
        elif "sp" in model_name:
            # from deeplab.dl_sp import get_model
            module = importlib.import_module("deeplab.dl_sp")
            cls = getattr(module, "get_model")
            return cls(checkpoint_path, self.num_classes)
        elif "mrb" in model_name:
            module = importlib.import_module("deeplab.dl_sp")
            cls = getattr(module, "get_model")
            return cls(checkpoint_path, self.num_classes)
        elif "all" in model_name:
            module = importlib.import_module("deeplab.dl_sp")
            cls = getattr(module, "get_model")
            return cls(checkpoint_path, self.num_classes)
        else:
            raise ValueError(f"Model {model_name} not found in the list of available models")
        # return get_model(checkpoint_path, self.num_classes)
    
    def _load_maskrcnn(self, model_name: str, checkpoint_path: str):
        from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
        if self.checkpoint_path is not None:
            return torch.load(checkpoint_path)
        model = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            model.roi_heads.box_predictor.cls_score.in_features, self.num_classes)
        model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
            model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, self.num_classes)
        return model
    
    def _load_unet(self, model_name: str, checkpoint_path: str):
        pass

def maskrcnn_prediction(pred: torch.Tensor)-> torch.Tensor:
    pred_masks = {1: [], 2: [], 3: []}
    target_masks = {1: [], 2: [], 3: []}
    # category_labels_rev = {
    #     1: "turtle",
    #     2: "flipper",
    #     3: "head",
    # }

    for i, label in enumerate(pred['labels']):
        cat = label.item()
        if cat in pred_masks:
            pred_masks[cat].append(pred['masks'][i] > 0.5)
    

    for cat, masks in pred_masks.items():
        if masks:
            pred_masks[cat] = torch.stack(masks).any(dim=0).float()
        else:
            pred_masks[cat] = torch.zeros_like(pred['masks'][0], dtype=torch.float32)

    # for cat, masks in target_masks.items():
    #     if masks:
    #         target_masks[cat] = torch.stack(masks).any(dim=0).float()
    #     else:
    #         target_masks[cat] = torch.zeros_like(pred['masks'][0], dtype=torch.float32)

    return pred_masks, target_masks

def import_necessary(model_name):
    if "deeplabv3" in model_name:
        from segmentation_models_pytorch import DeepLabV3Plus
    elif "sp" in model_name:
        from .deeplab.dl_sp import get_model, CustomizedDeeplabv3plus
    elif "mrb" in model_name:
        from .deeplab.dl_mrb import get_model, CustomizedDeeplabv3plus
    elif "all" in model_name:
        from .deeplab.dl_all import get_model, CustomizedDeeplabv3plus
    else:
        raise ValueError(f"Model {model_name} not found in the list of available models")
