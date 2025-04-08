import argparse
import logging
import math
import os
import random
import sys
import time
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate, evaluate_new
from unet import UNet
from unet.loss import LoggingCriterionModule, WeightedBinaryCrossEntropyLoss, WeightedBinaryCrossEntropyLossGlobal
from unet.unet_model_og import UNet_og
from utils.data_loading import BasicDataset, CarvanaDataset, GemsyDataset, GemsySplitLoader
from utils.dice_score import dice_loss

import albumentations as A
from albumentations.pytorch import ToTensorV2
#from focal_loss.focal_loss import FocalLoss
from torchvision.ops.focal_loss import sigmoid_focal_loss
#from sklearn.model_selection import KFold

#dir_img = Path('./data/imgs/')
dir_img = Path("C:\\Users\\Admin\\Desktop\\Gemsy\\Data\\processed\\.in\\features\\")
#dir_mask = Path('./data/masks/')
#dir_mask = Path("C:\\Users\\Admin\\Desktop\\Gemsy\\Data\\processed\\.in\\masks\\")
dir_mask = Path("C:\\Users\\Admin\\Desktop\\Gemsy\\Data\\processed\\.in\\adjusted_masks\\")
#dir_img = Path("C:\\Users\\Admin\\Desktop\\Gemsy\\Data\\processed\\.subsplit\\features\\")
#dir_mask = Path("C:\\Users\\Admin\\Desktop\\Gemsy\\Data\\processed\\.subsplit\\masks\\")

dir_checkpoint = Path('./checkpoints/')


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        features = ["gt"],
        train_defect_focus_rate = 0.7,
        patch_size = 512,
        patches_per_image = 300,
        restart_run = None,
        start_epoch = None,
        criterion_patch_size = 51,
        unet_type = "OG"
):
    training_augmentation = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.7),
       # A.ElasticTransform(p=0.3),
     #   A.RandomBrightnessContrast(p=0.3),
        #A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
     #   A.GridDistortion(p=0.2),
      #  A.CoarseDropout(max_height=16, max_width=16, max_holes=8, fill_value=0, p=0.3),
        ToTensorV2()
    ])
    
    # 0. Define Augmentation
    # training_augmentation = A.Compose([
    #         A.HorizontalFlip(p=0.5),
    #         A.VerticalFlip(p=0.5),
    #         A.RandomRotate90(p=0.5),
    #         A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    #       #  A.RandomBrightnessContrast(p=0.2),
    #         A.ElasticTransform(p=0.2),
    #       #  A.GaussNoise(p=0.2),
    #        # A.Normalize(),
    #         ToTensorV2()
    #     ])
    
    # 1. Create dataset
    try:
        patch_size = patch_size#512
        patches_per_image = patches_per_image #300
        #dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
        #dataset = GemsyPatchDataset(dir_img, dir_mask, img_scale, transform=None)#training_augmentation)
        #dataset = GemsyDataset(dir_img, dir_mask, patch_size, patches_per_image, img_scale, transform=training_augmentation)#training_augmentation)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    all_ids = GemsySplitLoader(dir_img, dir_mask).get_ids()
    n_val = int(len(all_ids) * val_percent)
    n_train = len(all_ids) - n_val

    # Additional params
    train_defect_focus_rate = train_defect_focus_rate

    train_ids, val_ids = random_split(all_ids, [n_train, n_val], generator=torch.Generator().manual_seed(1))
    train_set = GemsyDataset(dir_img, dir_mask, patch_size, patches_per_image, img_scale, features=features, transform=training_augmentation, ids=train_ids, defect_focus_rate=train_defect_focus_rate)
    val_set = GemsyDataset(dir_img, dir_mask, patch_size, patches_per_image*2, img_scale, features=features, transform=None, ids=val_ids, validation=True)

    n_val = n_val * patches_per_image
    n_train = n_train * patches_per_image
   # n_val = int(len(dataset) * val_percent)
   ## n_train = len(dataset) - n_val
   # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True) #os.cpu_count() - memory overload due to spawning 32 processes
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    #val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    if restart_run:
        experiment = wandb.init(project='U-Net', resume='must', anonymous='must', id=restart_run)#, id="zs97wupy") #resume=allow resume=must
    else:
        experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')#, id="zs97wupy") #resume=allow resume=must
    

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    #optimizer = optim.RMSprop(model.parameters(),
    #                          lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    
    #optimizer = optim.Adam(model.parameters(),
    #                          lr=learning_rate, weight_decay=weight_decay, foreach=True)

    optimizer = optim.SGD(model.parameters(),
                          lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.98)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # Binary Cross Entropy with Custom local Variance weighting and Dice loss
    criterion = LoggingCriterionModule(criterion_name="BCELV+D", patch_size=criterion_patch_size)
    #criterion = LoggingCriterionModule(criterion_name="BCE+D")

    if start_epoch and restart_run:
        start_epoch += 1
        global_step = math.ceil(n_train / (batch_size) * start_epoch)
    else:
        start_epoch = 0
        global_step = 0 

    if not restart_run:
        experiment.config.update(
            dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                 val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp,
                 momentum=momentum,weight_decay=weight_decay,
                 criterion=criterion.print_criterion(),
                 criterion_params=criterion.print_params(),
                 augmentations=str(training_augmentation) if training_augmentation else 'None',
                 dataset_features=features,
                 train_defect_focus_rate = train_defect_focus_rate,
                 train_ids = list(train_ids),
                 val_ids = list(val_ids),
                 patch_size = patch_size,
                 patches_per_image = patches_per_image,
                 scheduler = "MultiplicativeLR",
                 scheduler_params= {"lr_lambda epoch": 0.98},
                 optimizer="SGD",
                 optimizer_params={"lr": learning_rate, "weight_decay": weight_decay, "momentum":momentum},
                 non_linear_function="ReLu",
                 additional="Using Adjusted masks, variance applied to full",
                 unet_type=unet_type,
                 criterion_patch_size=criterion_patch_size
                 )
        )

    # 5. Begin training
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0
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

        iou = 0
        epoch_dice = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            batch_idx = 0
           # start_time = time.time()
            for batch in train_loader:
               # batch_time = (time.time() - start_time)
               # print(f"batch: {batch_time}")
                #start_time = time.time()

                batch_idx += 1
                images, true_masks = batch['image'], batch['mask']

                # i = 1
                # c = images[i, 6:9, ...].cpu().transpose(0, 2)
                # plt.figure(figsize = (5,5))
                # plt.imshow(c.cpu())

                # m = true_masks[i].cpu()
                # plt.figure(figsize = (5,5))
                # plt.imshow(m.cpu())

                # d = images[i, 6:9, ...].cpu().transpose(0, 2)
                # plt.figure(figsize = (5,5))
                # plt.imshow(d.cpu())

                # plt.close()
                # plt.close()

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                   # preprocess_time = (time.time() - start_time)
                   # start_time = time.time()

                    masks_pred = model(images)
                   # model_time = (time.time() - start_time)
                 #   start_time = time.time()

                    if model.n_classes == 1:
                        criterion_loss, dice_loss = criterion.get_loss(images, masks_pred, true_masks)
                        loss = criterion_loss + dice_loss
                        if torch.any(loss.isnan()) == True:
                            print("BAD TRAIN")
                     #   loss_time = (time.time() - start_time)
                     #   start_time = time.time()

                        pred_defected_patches = torch.any((F.sigmoid(masks_pred) > 0.5).float().squeeze(1), (1,2))
                        true_defected_patches = torch.any(true_masks, (1,2))

                        defected_patches += torch.sum(true_defected_patches).item()
                        non_defected_patches += torch.sum(torch.logical_not(true_defected_patches)).item()

                        TP_patches += torch.sum(torch.logical_and(pred_defected_patches, true_defected_patches)).item()
                        FP_patches += torch.sum(torch.logical_and(pred_defected_patches, torch.logical_not(true_defected_patches))).item()
                        FN_patches += torch.sum(torch.logical_and(torch.logical_not(pred_defected_patches), true_defected_patches)).item()

                        pred_defected_pxs = (F.sigmoid(masks_pred) > 0.5).float().squeeze(1)
                        true_defected_pxs = true_masks

                        defected_pxs += torch.sum(true_defected_pxs).item()
                        non_defected_pxs += torch.sum(torch.logical_not(true_defected_pxs)).item()

                        TP_pxs += torch.sum(torch.logical_and(pred_defected_pxs, true_defected_pxs)).item()
                        FP_pxs += torch.sum(torch.logical_and(pred_defected_pxs, torch.logical_not(true_defected_pxs))).item()
                        FN_pxs += torch.sum(torch.logical_and(torch.logical_not(pred_defected_pxs), true_defected_pxs)).item()
                    #    metrics_time = (time.time() - start_time)
                    #    start_time = time.time()

                    else:
                        raise Exception("Unimplemented multiple classes")
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

            #    important_time = (time.time() - start_time)
            #    start_time = time.time()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                epsilon = 1e-4
                train_recall_patch = TP_patches / (TP_patches+FN_patches+epsilon)
                train_accuracy_patch = TP_patches / (TP_patches+FP_patches+epsilon)
                defected_rate_patch = defected_patches / (defected_patches + non_defected_patches+epsilon)

                train_recall_px = TP_pxs / (TP_pxs+FN_pxs+epsilon)
                train_accuracy_px = TP_pxs / (TP_pxs+FP_pxs+epsilon)
                defected_rate_px = defected_pxs / (defected_pxs + non_defected_pxs+epsilon)

                iou = TP_pxs / (TP_pxs + FP_pxs + FN_pxs + epsilon)
                epoch_dice += dice_loss.item()

            #    metrics2_time = (time.time() - start_time)
            #    start_time = time.time()

                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch,
                    'epoch train patch Recall': train_recall_patch,
                    'epoch train patch Accuracy': train_accuracy_patch,
                    'epoch train defected patch rate': defected_rate_patch,
                    'epoch train pixel Recall': train_recall_px,
                    'epoch train pixel Accuracy': train_accuracy_px,
                    'epoch train defected pixel rate': defected_rate_px,
                    'epoch train IOU': iou,
                    'epoch train criterion loss': criterion_loss,
                    'epoch train DICE': epoch_dice / batch_idx,
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

            #    wandb_time = (time.time() - start_time)
             #   start_time = time.time()

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score, val_recall_patch, val_accuracy_patch, val_defected_rate_patch, val_recall_px, val_accuracy_px, val_defected_rate_px, val_iou = evaluate_new(model, val_set, device, amp) # val_loader
                        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):        
                            scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'validation patch Recall': val_recall_patch,
                                'validation patch Accuraccy': val_accuracy_patch,
                                'validation patch Defected_Rate': val_defected_rate_patch,
                                'validation pixel Recall': val_recall_px,
                                'validation pixel Accuraccy': val_accuracy_px,
                                'validation pixel Defected_Rate': val_defected_rate_px,
                                'validation IOU': val_iou,
                                'images': wandb.Image(images[0, :3, ...].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()) if model.n_classes > 1 else wandb.Image((F.sigmoid(masks_pred))[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass
                #validation_time = (time.time() - start_time)
               # start_time = time.time()
                #print(f"preproceess: {preprocess_time}, \n model:{model_time}, \n loss: {loss_time}, \n metrics: {metrics_time}, \n important: {important_time}, \n metrics2: {metrics2_time}, \n wandb: {wandb_time}, \n validation: {validation_time}")

        if isinstance(scheduler, torch.optim.lr_scheduler.MultiplicativeLR):        
            scheduler.step()

        if save_checkpoint:
            
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = train_set.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')




def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=180, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.03, #0.03, #5e-6, 0.03
                        help='Learning rate', dest='lr')
    parser.add_argument('--weight_decay', '-wd', metavar='WD', type=float, default=0, #1e-8, 
                        help='Weight Decay', dest='wd')
    parser.add_argument('--momentum', '-m', metavar='M', type=float, default=0.95,  #0.999
                        help='Momentum', dest='m')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file') 
    #parser.add_argument('--load', '-f', type=str, default="C:\\Users\\Admin\\Desktop\\Gemsy\\External\\Pytorch-UNet\\checkpoints\\checkpoint_epoch1.pth", help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=25.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(filename="./log.txt", level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')


    train_defect_focus_rate = 0.7
    patch_size = 512
    patches_per_image = 300 # 300
    criterion_patch_size = 45

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    features = ["full", "dbscantuned", "opacity"]
    n_channels = len(features) * 3
    unet_type = "OG"

    if unet_type == "OG":
        model = UNet_og(n_channels=n_channels, n_classes=args.classes, bilinear=args.bilinear)
    else:
        model = UNet(n_channels=n_channels, n_classes=args.classes, bilinear=args.bilinear)
    
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    start_epoch = None
    restart_run = None
   # start_epoch = 1
    #restart_run = "tsiltzmj"

    if args.load:
        args.load
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')
        #start_epoch = 0  #INSEERT
       # restart_run = "id" #INSERT


    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            weight_decay=args.wd,
            momentum=args.m,
            features = features,
            unet_type = unet_type,
            start_epoch=start_epoch,
            restart_run=restart_run,
            train_defect_focus_rate = train_defect_focus_rate,
            patch_size = patch_size,
            patches_per_image = patches_per_image,
            criterion_patch_size = criterion_patch_size,
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            weight_decay=args.wd,
            momentum=args.m,
            features=features
        )
