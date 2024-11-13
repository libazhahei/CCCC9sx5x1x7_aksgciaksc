import torch
import torchvision
from typing import Callable
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
            return self._load_maskrcnn(model_name, checkpoint_path), lambda x: x['masks']
        elif "unet" in model_name:
            return self._load_unet(model_name, checkpoint_path), lambda x: x
        pass 

    def _load_deeplab_models(self, model_name: str, checkpoint_path: str):
        if "baseline" in model_name:
            import segmentation_models_pytorch as smp
            if checkpoint_path is None:
                return smp.DeepLabV3Plus(
                    encoder_name='resnet101', 
                    encoder_weights=None, 
                    classes=4, 
                    activation='softmax2d',
                )
            else:
                return torch.load(checkpoint_path)
        elif "sp" in model_name:
            from deeplab.dl_sp import CustomizedDeeplabv3plus
        elif "mrb" in model_name:
            from deeplab.dl_mrb import CustomizedDeeplabv3plus
        elif "all" in model_name:
            from deeplab.dl_all import CustomizedDeeplabv3plus
        else:
            raise ValueError(f"Model {model_name} not found in the list of available models")
        if checkpoint_path is None:
            return CustomizedDeeplabv3plus(
                encoder_name='resnet101', 
                encoder_weights=None, 
                classes=self.num_classes, 
                activation='softmax2d',
            )
        else:
            return torch.load(checkpoint_path)
    
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

