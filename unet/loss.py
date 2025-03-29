import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale

from utils.dice_score import dice_loss

# Python implementation of local-variance BCE loss provided by paper:
# "
#  Surface Defect Detection for Mobile Phone Back Glass Based on 
#  Symmetric Convolutional Neural Network Deep Learning
# "
def compute_local_variance(image, patch_size):
    padding = patch_size // 2 # patch size must be uneven to center around single pixel
    # Mean of each pixel in a patch (patch size defines local area size)
    mean = F.avg_pool2d(image, kernel_size=patch_size, stride=1, padding=padding)
    # Mean of squares
    mean_sq = F.avg_pool2d(image**2, kernel_size=patch_size, stride=1, padding=padding)
    # Variance formula
    variance = mean_sq - mean**2
    return variance

def compute_global_variance(image):
    mean = image.mean(dim=(-1, -2), keepdim=True)  # Mean over H and W
    mean_sq = (image ** 2).mean(dim=(-1, -2), keepdim=True)  # Mean of squares
    variance = mean_sq - mean ** 2
    return variance

def compute_weight_map(variance_map, bias=1e-4):
    weight_map = variance_map + bias
    # Normalize across spatial dimensions
    weight_map_sum = weight_map.sum(dim=(-1, -2), keepdim=True)
    weight_map = weight_map / weight_map_sum
    return weight_map

def weighted_bce_loss(logits, ground_truth, weight_map, eps=1e-8):
    prediction = F.sigmoid(logits)
    prediction = torch.clamp(prediction, eps, 1 - eps)  # Avoid log(0)
    loss = - (ground_truth * torch.log(prediction) + (1 - ground_truth) * torch.log(1 - prediction))
    weighted_loss = (loss * weight_map).sum()
    return weighted_loss

class WeightedBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, bce_eps=1e-8, variance_bias=1e-4, patch_size=11): #variance_bias=1e-8 poroduces nans
        super().__init__()
        self.eps = bce_eps
        self.bias = variance_bias
        self.patch_size = patch_size

    def forward(self, images, logits, target):
        grayscale = rgb_to_grayscale(images)
        variance_map = compute_local_variance(grayscale, self.patch_size)
        weight_map = compute_weight_map(variance_map, self.bias)
        binary_loss = weighted_bce_loss(logits, target, weight_map, self.eps)
        return binary_loss
    
class WeightedBinaryCrossEntropyLossGlobal(nn.Module):
    def __init__(self, bce_eps=1e-8, variance_bias=1e-4):
        super().__init__()
        self.eps = bce_eps
        self.bias = variance_bias

    def forward(self, images, logits, target):
        grayscale = rgb_to_grayscale(images)
        variance_map = compute_global_variance(grayscale)
        weight_map = compute_weight_map(variance_map, self.bias)
        binary_loss = weighted_bce_loss(logits, target, weight_map, self.eps)
        return binary_loss
    
class LoggingCriterionModule():
    def __init__(self, criterion_name="BCELV", **kwargs):
        self.criterion_name =  criterion_name
        self.criterion = None
        self.get_loss = None
        self.kwargs = kwargs

        if criterion_name == "BCELV":
            self.criterion = WeightedBinaryCrossEntropyLoss(**self.kwargs)
            self.get_loss = self.BCELV_loss
        elif criterion_name == "BCELV+D":
            self.criterion = WeightedBinaryCrossEntropyLoss(**self.kwargs)
            self.get_loss = self.BCELV_D_loss

    def BCELV_loss(self, images, masks_pred, true_masks):
        if images.shape[1] > 3:
            images = images[:, :3, :, :]
        masks_pred = masks_pred.squeeze(1)
        true_masks = true_masks.float()
        loss = self.criterion(images, masks_pred, true_masks, self.kwargs)
        return loss
    
    def BCELV_D_loss(self, images, masks_pred, true_masks):
        if images.shape[1] > 3:
            images = images[:, :3, :, :]
        masks_pred = masks_pred.squeeze(1)
        true_masks = true_masks.float()
        loss_crit = self.criterion(images, masks_pred, true_masks)
        loss_dice = dice_loss(F.sigmoid(masks_pred), true_masks, multiclass=False)
        return loss_crit, loss_dice
    
    # Criterion part
    #criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    #criterion = WeightedBinaryCrossEntropyLoss(patch_size=31)
    #criterion = WeightedBinaryCrossEntropyLossGlobal()
    #criterion = sigmoid_focal_loss #FocalLoss(gamma=0.7)
    #    defective = 6813187
    #non_defective = 1088686589
    #pos_weight = torch.tensor([non_defective / defective], device="cuda")  # ≈ 159.7
    #criterion = nn.BCEWithLogitsLoss()#pos_weight=pos_weight)

# Loss part
 # loss = criterion(images, masks_pred.squeeze(1), true_masks.float()) # variance-based
                        #loss = criterion(masks_pred.squeeze(1), true_masks.float()) # traditional
                       # dice_val = dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                       # loss += dice_val
                        #loss = criterion(masks_pred.squeeze(1), true_masks.float(), reduction='mean')
                        #loss = dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
    def print_params(self):
        return str(self.kwargs)
    
    def print_criterion(self):
        return self.criterion_name
