import torch
import torch.nn as nn

class MultiClassIoU(nn.Module):
    def __init__(self, num_classes, ignore_index=None, eps=1e-6, single_calss_id: tuple[int, str]=None):
        super(MultiClassIoU, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps
        self.single_calss_id = single_calss_id

    @property
    def __name__(self):
        if self.single_calss_id is not None:
            return f"Class_{self.single_calss_id[1]}_IoU"
        return "MultiClassIoU"

    def __repr__(self):
        return f"MultiClassIoU(num_classes={self.num_classes})"

    def __str__(self):
        return f"MultiClassIoU(num_classes={self.num_classes})"

    def forward(self, predictions, targets):
        # return self.per_class_iou(predictions, targets)
        device = predictions.device
        if len(predictions.shape) == 4:  # (B, C, H, W)
            predictions = torch.argmax(predictions, dim=1)  # Convert to (B, H, W)

        predictions = predictions.to(device)
        targets = targets.to(device)

        iou_per_class = []

        for class_idx in range(self.num_classes):
            if class_idx == self.ignore_index:
                continue
            pred_mask = (predictions == class_idx).float()
            target_mask = (targets == class_idx).float()
            intersection = (pred_mask * target_mask).sum().to(device)
            union = (pred_mask + target_mask).gt(0).float().sum().to(device)
            iou = torch.where(
                union > 0,
                intersection / (union + self.eps),
                torch.tensor(0.0, device=device)
            )

            iou_per_class.append(iou)

        if self.single_calss_id is not None:
            return iou_per_class[self.single_calss_id[0]]

        if len(iou_per_class) > 0:
            iou_per_class = torch.stack(iou_per_class)
            mean_iou = iou_per_class.mean()
        else:
            mean_iou = torch.tensor(1e-6, device=device)

        return mean_iou

    def per_class_iou(self, predictions, targets):
        device = predictions.device
        if len(predictions.shape) == 4:
            predictions = torch.argmax(predictions, dim=1)

        predictions = predictions.to(device)
        targets = targets.to(device)

        results = {}

        for class_idx in range(self.num_classes):
            if class_idx == self.ignore_index:
                continue

            pred_mask = (predictions == class_idx).float()
            target_mask = (targets == class_idx).float()

            intersection = (pred_mask * target_mask).sum().to(device)
            union = (pred_mask + target_mask).gt(0).float().sum().to(device)

            iou = torch.where(
                union > 0,
                intersection / (union + self.eps),
                torch.tensor(0.0, device=device)
            )

            results[f'class_{class_idx}_iou'] = iou.item()

        return results

class MultiClassF1Score(nn.Module):
    def __init__(self, num_classes, epsilon=1e-7, reduction='mean', single_calss_id: tuple[int, str]=None):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.single_calss_id = single_calss_id

    def forward(self, predictions, targets):
        predictions = torch.argmax(predictions, dim=1)
        f1_scores = []

        for class_idx in range(self.num_classes):
            pred_mask = (predictions == class_idx)
            true_mask = (targets == class_idx)
            true_positive = torch.sum(pred_mask & true_mask).float()
            false_positive = torch.sum(pred_mask & ~true_mask).float()
            false_negative = torch.sum(~pred_mask & true_mask).float()
            precision = true_positive / (true_positive + false_positive + self.epsilon)
            recall = true_positive / (true_positive + false_negative + self.epsilon)
            f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
            f1_scores.append(f1)
        f1_scores = torch.tensor(f1_scores, device=predictions.device)
        if self.single_calss_id is not None:
            return f1_scores[self.single_calss_id[0]]
        if self.reduction == 'none':
            return f1_scores
        elif self.reduction == 'mean':
            return torch.mean(f1_scores)
        elif self.reduction == 'sum':
            return torch.sum(f1_scores)
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

    def extra_repr(self) -> str:
        if self.single_calss_id is not None:
            return f'F1 socre class={self.single_calss_id[1]}'
        return f'F1 socre num_classes={self.num_classes}, reduction={self.reduction}'
    

class MultiClassPrecision(nn.Module):
    def __init__(self, num_classes, epsilon=1e-7, reduction='mean', single_calss_id: tuple[int, str]=None):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.single_calss_id = single_calss_id

    def forward(self, predictions, targets):
        predictions = torch.argmax(predictions, dim=1)
        precision_scores = []

        for class_idx in range(self.num_classes):
            pred_mask = (predictions == class_idx)
            true_mask = (targets == class_idx)
            true_positive = torch.sum(pred_mask & true_mask).float()
            false_positive = torch.sum(pred_mask & ~true_mask).float()
            precision = true_positive / (true_positive + false_positive + self.epsilon)
            precision_scores.append(precision)
        precision_scores = torch.tensor(precision_scores, device=predictions.device)
        if self.single_calss_id is not None:
            return precision_scores[self.single_calss_id[0]]
        if self.reduction == 'none':
            return precision_scores
        elif self.reduction == 'mean':
            return torch.mean(precision_scores)
        elif self.reduction == 'sum':
            return torch.sum(precision_scores)
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

    def extra_repr(self) -> str:
        if self.single_calss_id is not None:
            return f'Precision class={self.single_calss_id[1]}'
        return f'Precision num_classes={self.num_classes}, reduction={self.reduction}'

class MultiClassRecall(nn.Module):
    def __init__(self, num_classes, epsilon=1e-7, reduction='mean', single_calss_id: tuple[int, str]=None):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.single_calss_id = single_calss_id

    def forward(self, predictions, targets):
        predictions = torch.argmax(predictions, dim=1)
        recall_scores = []

        for class_idx in range(self.num_classes):
            pred_mask = (predictions == class_idx)
            true_mask = (targets == class_idx)
            true_positive = torch.sum(pred_mask & true_mask).float()
            false_negative = torch.sum(~pred_mask & true_mask).float()
            recall = true_positive / (true_positive + false_negative + self.epsilon)
            recall_scores.append(recall)
        recall_scores = torch.tensor(recall_scores, device=predictions.device)
        if self.single_calss_id is not None:
            return recall_scores[self.single_calss_id[0]]
        if self.reduction == 'none':
            return recall_scores
        elif self.reduction == 'mean':
            return torch.mean(recall_scores)
        elif self.reduction == 'sum':
            return torch.sum(recall_scores)
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

    def extra_repr(self) -> str:
        if self.single_calss_id is not None:
            return f'Recall class={self.single_calss_id[1]}'
        return f'num_classes={self.num_classes}, reduction={self.reduction}'