import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    TP = 0
    FP = 0
    FN = 0

    defected = 0
    non_defected = 0
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
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float().squeeze(1)

                pred_defects = torch.any(mask_pred, (1,2))
                true_defects = torch.any(mask_true, (1,2))

                defected += torch.sum(true_defects).item()
                non_defected += torch.sum(torch.logical_not(true_defects)).item()

                TP += torch.sum(torch.logical_and(pred_defects, true_defects)).item()
                FP += torch.sum(torch.logical_and(pred_defects, torch.logical_not(true_defects))).item()
                FN += torch.sum(torch.logical_and(torch.logical_not(pred_defects), true_defects)).item()

                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
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
    val_recall = TP / (TP+FN+epsilon)
    val_accuracy = TP / (TP+FP+epsilon)
    defected_rate = defected / (non_defected+epsilon)
    return dice_score / max(num_val_batches, 1), val_recall, val_accuracy, defected_rate
