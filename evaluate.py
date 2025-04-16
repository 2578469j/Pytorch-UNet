import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from utils.data_loading import GemsyDataset
from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    TP_patches = 0
    FP_patches = 0
    FN_patches = 0
    TP_pxs = 0
    FP_pxs = 0
    FN_pxs = 0
    defected_patches = 0
    non_defected_patches = 0
    defected_pxs = 0
    non_defected_pxs = 0

    val_iou = 0
    dice_score = 0
    
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                #mask_pred = (F.sigmoid(mask_pred) > 0.5).float().squeeze(1)

                pred_defected_patches = torch.any((F.sigmoid(mask_pred) > 0.5).float().squeeze(1), (1,2))
                true_defected_patches = torch.any(mask_true, (1,2))

                defected_patches += torch.sum(true_defected_patches).item()
                non_defected_patches += torch.sum(torch.logical_not(true_defected_patches)).item()

                TP_patches += torch.sum(torch.logical_and(pred_defected_patches, true_defected_patches)).item()
                FP_patches += torch.sum(torch.logical_and(pred_defected_patches, torch.logical_not(true_defected_patches))).item()
                FN_patches += torch.sum(torch.logical_and(torch.logical_not(pred_defected_patches), true_defected_patches)).item()

                pred_defected_pxs = (F.sigmoid(mask_pred) > 0.5).float().squeeze(1)
                true_defected_pxs = mask_true

                defected_pxs += torch.sum(true_defected_pxs).item()
                non_defected_pxs += torch.sum(torch.logical_not(true_defected_pxs)).item()

                TP_pxs += torch.sum(torch.logical_and(pred_defected_pxs, true_defected_pxs)).item()
                FP_pxs += torch.sum(torch.logical_and(pred_defected_pxs, torch.logical_not(true_defected_pxs))).item()
                FN_pxs += torch.sum(torch.logical_and(torch.logical_not(pred_defected_pxs), true_defected_pxs)).item()


                # compute the Dice score
                dice_score += dice_coeff((F.sigmoid(mask_pred) > 0.5).float().squeeze(1), mask_true, reduce_batch_first=False)
                if torch.any(dice_score.isnan()) == True:
                    print("BAD VALIDATION")

                torch.any(mask_true) # B, C, W, H

            else:
                # TODO maybe future multi-class
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    epsilon = 1e-4
    val_recall_patch = TP_patches / (TP_patches+FN_patches+epsilon)
    val_accuracy_patch = TP_patches / (TP_patches+FP_patches+epsilon)
    val_defected_rate_patch = defected_patches / (defected_patches + non_defected_patches+epsilon)

    val_recall_px = TP_pxs / (TP_pxs+FN_pxs+epsilon)
    val_accuracy_px = TP_pxs / (TP_pxs+FP_pxs+epsilon)
    val_defected_rate_px = defected_pxs / (defected_pxs + non_defected_pxs+epsilon)

    val_iou = TP_pxs / (TP_pxs + FP_pxs + FN_pxs + epsilon)
    return dice_score / max(num_val_batches, 1), val_recall_patch, val_accuracy_patch, val_defected_rate_patch, val_recall_px, val_accuracy_px, val_defected_rate_px, val_iou

def unpatchify_predictions(patches, mask_patches_gt, img_shape, patch_size, overlap, avg=False):
    """
    patches: torch.Tensor of shape (N, 1, H, W) or (N, H, W)
    masks_gt: torch.Tensor of shape (N, 1, H, W) or (N, H, W)
    img_shape: tuple (H, W) — original image shape
    patch_size: int — size of each patch (assumes square patches)
    overlap: float — overlap used during patching (0 to <1)

    Returns:
        torch.Tensor of shape img_shape (single-channel mask)
    """
    H, W = img_shape
    out_shape = (1, H, W)
    step = int(patch_size * (1 - overlap))
    result = torch.zeros(out_shape, dtype=torch.float32)
    mask = torch.zeros(out_shape, dtype=torch.float32)
    count = torch.zeros(out_shape, dtype=torch.float32)

    idx = 0
    for y in range(0, H - patch_size + 1, step):
        for x in range(0, W - patch_size + 1, step):
            patch = patches[idx]
            mask_patch = mask_patches_gt[idx]
            if patch.ndim == 2:  # If (H, W)
                patch = patch.unsqueeze(0)  # -> (1, H, W)
            if mask_patch.ndim == 2:
                mask_patch = mask_patch.unsqueeze(0)

            result[:, y:y+patch_size, x:x+patch_size] += patch
            mask[:, y:y+patch_size, x:x+patch_size] += mask_patch
            count[:, y:y+patch_size, x:x+patch_size] += 1
            idx += 1

    # Avoid divide by zero
    count[count == 0] = 1
    if avg:
        result /= count
    else:
        result = np.clip(result, a_min=0, a_max=1)
    mask = np.clip(mask, a_min=0, a_max=1)
    return result, mask

@torch.inference_mode()
def evaluate_new(net, dataset:GemsyDataset, device, amp, batch_size=4, out_threshold=0.5, overlap=0):
    net.eval()
    num_val_batches = len(dataset.ids)
    TP_patches = 0
    FP_patches = 0
    FN_patches = 0
    TP_pxs = 0
    FP_pxs = 0
    FN_pxs = 0
    defected_patches = 0
    non_defected_patches = 0
    defected_pxs = 0
    non_defected_pxs = 0

    val_iou = 0
    dice_score = 0
    
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for idx in tqdm(range(num_val_batches), total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            stacked_img_patches, og_size = dataset.get_prediction_data(idx)

            # move images and labels to correct device and type
            mask_patches_gt = stacked_img_patches[:, :1].to(device=device, dtype=torch.long, memory_format=torch.channels_last).squeeze(1)

            stacked_img_patches = stacked_img_patches[:, 1:].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

            outputs = []
            for i in range(0, stacked_img_patches.shape[0], batch_size):
                batch = stacked_img_patches[i:i+batch_size]
                mask_true = mask_patches_gt[i:i+batch_size]
                # predict the mask
                mask_pred_logits = net(batch)

                if net.n_classes == 1:
                    assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                    mask_pred = (F.sigmoid(mask_pred_logits) > 0.5).float().squeeze(1)
    
                    pred_defected_patches = torch.any(mask_pred, (1,2))
                    true_defected_patches = torch.any(mask_true, (1,2))
    
                    defected_patches += torch.sum(true_defected_patches).item()
                    non_defected_patches += torch.sum(torch.logical_not(true_defected_patches)).item()
    
                    TP_patches += torch.sum(torch.logical_and(pred_defected_patches, true_defected_patches)).item()
                    FP_patches += torch.sum(torch.logical_and(pred_defected_patches, torch.logical_not(true_defected_patches))).item()
                    FN_patches += torch.sum(torch.logical_and(torch.logical_not(pred_defected_patches), true_defected_patches)).item()
                else:
                    raise Exception("Unimplemented multiple classes")

                    # TODO maybe future multi-class
                    assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                    # convert to one-hot format
                    mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                    mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                    # compute the Dice score, ignoring background
                    dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                outputs.append(mask_pred)
            
            patch_outputs = torch.cat(outputs, dim=0).to("cpu")
            pred_defected_pxs, true_defected_pxs = unpatchify_predictions(patch_outputs, mask_patches_gt.to("cpu"), (og_size[-2], og_size[-1]), patch_size=dataset.patch_size, overlap=overlap)

            defected_pxs += torch.sum(true_defected_pxs).item()
            non_defected_pxs += torch.sum(torch.logical_not(true_defected_pxs)).item()
            TP_pxs += torch.sum(torch.logical_and(pred_defected_pxs, true_defected_pxs)).item()
            FP_pxs += torch.sum(torch.logical_and(pred_defected_pxs, torch.logical_not(true_defected_pxs))).item()
            FN_pxs += torch.sum(torch.logical_and(torch.logical_not(pred_defected_pxs), true_defected_pxs)).item()
            
            # compute the Dice score
            dice_score += dice_coeff((F.sigmoid(mask_pred) > 0.5).float().squeeze(1), mask_true, reduce_batch_first=False)
            if torch.any(dice_score.isnan()) == True:
                print("BAD VALIDATION")

           

    net.train()
    epsilon = 1e-4
    val_recall_patch = TP_patches / (TP_patches+FN_patches+epsilon)
    val_accuracy_patch = TP_patches / (TP_patches+FP_patches+epsilon)
    val_defected_rate_patch = defected_patches / (defected_patches + non_defected_patches+epsilon)

    val_recall_px = TP_pxs / (TP_pxs+FN_pxs+epsilon)
    val_accuracy_px = TP_pxs / (TP_pxs+FP_pxs+epsilon)
    val_defected_rate_px = defected_pxs / (defected_pxs + non_defected_pxs+epsilon)

    val_iou = TP_pxs / (TP_pxs + FP_pxs + FN_pxs + epsilon)
    return dice_score / max(num_val_batches, 1), val_recall_patch, val_accuracy_patch, val_defected_rate_patch, val_recall_px, val_accuracy_px, val_defected_rate_px, val_iou

@torch.inference_mode()
def evaluate_direct_mask(net, dataset:GemsyDataset, device, amp, batch_size=4, out_threshold=0.5, overlap=0):
    net.eval()
    num_val_batches = len(dataset.ids)
    TP_patches = 0
    FP_patches = 0
    FN_patches = 0
    TP_pxs = 0
    FP_pxs = 0
    FN_pxs = 0
    defected_patches = 0
    non_defected_patches = 0
    defected_pxs = 0
    non_defected_pxs = 0

    val_iou = 0
    dice_score = 0
    
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for idx in tqdm(range(num_val_batches), total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            stacked_img_patches, og_size = dataset.get_prediction_data(idx)

            # move images and labels to correct device and type
            mask_patches_gt = stacked_img_patches[:, :1].to(device=device, dtype=torch.long, memory_format=torch.channels_last).squeeze(1)

            stacked_img_patches = stacked_img_patches[:, 1:].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

            outputs = []
            for i in range(0, stacked_img_patches.shape[0], batch_size):
                batch = stacked_img_patches[i:i+batch_size]
                mask_true = mask_patches_gt[i:i+batch_size]
                # predict the mask
                mask_pred_logits = torchvision.transforms.functional.rgb_to_grayscale(batch) #net(batch)

                if net.n_classes == 1:
                    assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                    mask_pred = (F.sigmoid(mask_pred_logits) > 0.5).float().squeeze(1)
    
                    pred_defected_patches = torch.any(mask_pred, (1,2))
                    true_defected_patches = torch.any(mask_true, (1,2))
    
                    defected_patches += torch.sum(true_defected_patches).item()
                    non_defected_patches += torch.sum(torch.logical_not(true_defected_patches)).item()
    
                    TP_patches += torch.sum(torch.logical_and(pred_defected_patches, true_defected_patches)).item()
                    FP_patches += torch.sum(torch.logical_and(pred_defected_patches, torch.logical_not(true_defected_patches))).item()
                    FN_patches += torch.sum(torch.logical_and(torch.logical_not(pred_defected_patches), true_defected_patches)).item()
                else:
                    raise Exception("Unimplemented multiple classes")

                    # TODO maybe future multi-class
                    assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                    # convert to one-hot format
                    mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                    mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                    # compute the Dice score, ignoring background
                    dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                outputs.append(mask_pred)
            
            patch_outputs = torch.cat(outputs, dim=0).to("cpu")
            pred_defected_pxs, true_defected_pxs = unpatchify_predictions(patch_outputs, mask_patches_gt.to("cpu"), (og_size[-2], og_size[-1]), patch_size=dataset.patch_size, overlap=overlap)

            defected_pxs += torch.sum(true_defected_pxs).item()
            non_defected_pxs += torch.sum(torch.logical_not(true_defected_pxs)).item()
            TP_pxs += torch.sum(torch.logical_and(pred_defected_pxs, true_defected_pxs)).item()
            FP_pxs += torch.sum(torch.logical_and(pred_defected_pxs, torch.logical_not(true_defected_pxs))).item()
            FN_pxs += torch.sum(torch.logical_and(torch.logical_not(pred_defected_pxs), true_defected_pxs)).item()
            
            # compute the Dice score
            dice_score += dice_coeff((F.sigmoid(mask_pred) > 0.5).float().squeeze(1), mask_true, reduce_batch_first=False)
            if torch.any(dice_score.isnan()) == True:
                print("BAD VALIDATION")

           

    net.train()
    epsilon = 1e-4
    val_recall_patch = TP_patches / (TP_patches+FN_patches+epsilon)
    val_accuracy_patch = TP_patches / (TP_patches+FP_patches+epsilon)
    val_defected_rate_patch = defected_patches / (defected_patches + non_defected_patches+epsilon)

    val_recall_px = TP_pxs / (TP_pxs+FN_pxs+epsilon)
    val_accuracy_px = TP_pxs / (TP_pxs+FP_pxs+epsilon)
    val_defected_rate_px = defected_pxs / (defected_pxs + non_defected_pxs+epsilon)

    val_iou = TP_pxs / (TP_pxs + FP_pxs + FN_pxs + epsilon)
    return dice_score / max(num_val_batches, 1), val_recall_patch, val_accuracy_patch, val_defected_rate_patch, val_recall_px, val_accuracy_px, val_defected_rate_px, val_iou
