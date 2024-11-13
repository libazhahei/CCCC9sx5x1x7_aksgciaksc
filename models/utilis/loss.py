import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiClassCombinedLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 alpha=0.5,    # Weight for Dice Loss
                 beta=0.3,     # Weight for Focal Loss
                 gamma=0.2,    # Weight for Cross Entropy
                 focal_gamma=2, # Focal Loss focusing parameter
                 epsilon=1e-6,  # Small constant to avoid division by zero
                 class_weights=None):

        super(MultiClassCombinedLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.focal_gamma = focal_gamma
        self.epsilon = epsilon
        self.name = "MultiClassIoU"

        if class_weights is None:
            self.class_weights = torch.ones(num_classes)
        else:
            self.class_weights = class_weights

    def dice_loss(self, inputs, targets):
        batch_size = inputs.size(0)
        inputs = F.softmax(inputs, dim=1)
        inputs = inputs.reshape(batch_size, self.num_classes, -1)
        targets = targets.reshape(batch_size, -1)
        dice_loss = 0.0
        for class_idx in range(self.num_classes):
            inputs_class = inputs[:, class_idx, :]  # (B, H*W)
            targets_class = (targets == class_idx).float()  # (B, H*W)
            intersection = (inputs_class * targets_class).sum(dim=1)  # (B,)
            union = inputs_class.sum(dim=1) + targets_class.sum(dim=1)  # (B,)
            class_dice = (2. * intersection + self.epsilon) / (union + self.epsilon)  # (B,)
            class_dice = class_dice.mean()
            dice_loss += (1 - class_dice) * self.class_weights[class_idx].to(inputs.device)
        return dice_loss / self.num_classes



    def focal_loss(self, inputs, targets):
        eps = 1e-7
        inputs = inputs.reshape(-1, self.num_classes)
        targets = targets.reshape(-1).long()  # Ensure targets are long type
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1.)
        probs = torch.clamp(probs, eps, 1. - eps)
        ce = -(targets_one_hot * torch.log(probs)).sum(dim=1)
        pt = (targets_one_hot * probs).sum(dim=1)
        focal_term = (1 - pt) ** self.focal_gamma
        focal_loss = focal_term * ce
        if self.class_weights is not None:
            class_weights = self.class_weights.to(inputs.device)
            weight_term = (targets_one_hot * class_weights.unsqueeze(0)).sum(dim=1)
            focal_loss = focal_loss * weight_term
        return focal_loss.mean()

    def cross_entropy_loss(self, inputs, targets):
        inputs = inputs.reshape(-1, self.num_classes)
        targets = targets.reshape(-1)
        log_probs = F.log_softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        weights = self.class_weights.to(inputs.device)
        weighted_targets = targets_one_hot * weights.unsqueeze(0)
        ce_loss = -(weighted_targets * log_probs).sum(dim=1).mean()
        return ce_loss

    def forward(self, inputs, targets):
        inputs = inputs.float()
        targets = targets.long()
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        ce = self.cross_entropy_loss(inputs, targets)
        combined_loss = (self.alpha * dice +
                        self.beta * focal +
                        self.gamma * ce)
        return combined_loss

    @property
    def __name__(self):
        return "MultiClassCombinedLoss"

    def __repr__(self):
        return f"MultiClassCombinedLoss(num_classes={self.num_classes})"

    def __str__(self):
        return f"MultiClassCombinedLoss(num_classes={self.num_classes})"