import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale

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

def compute_weight_map(variance_map, bias=1e-4):
    weight_map = variance_map + bias
    # Normalize across spatial dimensions
    weight_map_sum = weight_map.sum(dim=(2, 3), keepdim=True)
    weight_map = weight_map / (weight_map_sum + bias)
    return weight_map

def weighted_bce_loss(logits, ground_truth, weight_map, eps=1e-4):
    prediction = F.sigmoid(logits)
    prediction = torch.clamp(prediction, eps, 1 - eps)  # Avoid log(0)
    loss = - (ground_truth * torch.log(prediction) + (1 - ground_truth) * torch.log(1 - prediction))
    weighted_loss = (loss * weight_map).sum()
    return weighted_loss

class WeightedBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, bce_eps=1e-4, variance_bias=1e-4, patch_size=11):
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