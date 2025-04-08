import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet.unet_model_og import UNet_og
from utils.data_loading import GemsyDataset
from unet import UNet
from utils.utils import plot_img_and_mask, plot_img_and_mask_binary

dir_img = Path("C:\\Users\\Admin\\Desktop\\Gemsy\\Data\\processed\\.in\\features\\")
dir_mask = Path("C:\\Users\\Admin\\Desktop\\Gemsy\\Data\\processed\\.in\\masks\\")
patch_size = 512
patches_per_image = 350
img_scale = 0.5

def unpatchify_predictions(patches, img_shape, patch_size, overlap, avg=False):
    """
    patches: torch.Tensor of shape (N, 1, H, W) or (N, H, W)
    img_shape: tuple (C, H, W) — original image shape
    patch_size: int — size of each patch (assumes square patches)
    overlap: float — overlap used during patching (0 to <1)

    Returns:
        torch.Tensor of shape img_shape (single-channel mask)
    """
    _, H, W = img_shape
    step = int(patch_size * (1 - overlap))
    result = torch.zeros(img_shape, dtype=torch.float32)
    count = torch.zeros(img_shape, dtype=torch.float32)

    idx = 0
    for y in range(0, H - patch_size + 1, step):
        for x in range(0, W - patch_size + 1, step):
            patch = patches[idx]
            if patch.ndim == 2:  # If (H, W)
                patch = patch.unsqueeze(0)  # -> (1, H, W)
            result[:, y:y+patch_size, x:x+patch_size] += patch
            count[:, y:y+patch_size, x:x+patch_size] += 1
            idx += 1

    # Avoid divide by zero
    count[count == 0] = 1
    if avg:
        result /= count
    else:
        result = np.clip(result, a_min=0, a_max=1)
    return result

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5,
                features=['gt'],
                overlap = 0):
    net.eval()
    gset = GemsyDataset(dir_img, dir_mask, patch_size, patches_per_image, img_scale, features=features)
    stacked_img, og_size, og_img, mask_gt = gset.get_prediction_data_predict(full_img, overlap=overlap)
    #img = torch.from_numpy(GemsyDataset.preprocess(None, full_img, scale_factor, is_mask=False))
   # stacked_img = stacked_img.unsqueeze(0)
    #img = img.to(device=device, dtype=torch.float32)
    img = stacked_img.to(device=device, dtype=torch.float32)
    batch_size = 2
    outputs = []
    with torch.no_grad():
        for i in range(0, img.shape[0], batch_size):
            batch = img[i:i+batch_size]
            out = net(batch)
            #output = F.interpolate(out, (full_img.size[1], full_img.size[0]), mode='bilinear').cpu()
            if net.n_classes > 1:
                mask = out.argmax(dim=1)
            else:
                mask = torch.sigmoid(out) > out_threshold

            outputs.append(mask.cpu())

    patch_outputs = torch.cat(outputs, dim=0)  # [B, C, H, W]
    mask = unpatchify_predictions(patch_outputs, (1, og_size[-2], og_size[-1]), patch_size=gset.patch_size, overlap=overlap)
    #full_img = unpatchify_predictions(stacked_img.cpu()[:, :3, ...], (3, og_size[-2], og_size[-1]), patch_size=gset.patch_size, overlap=overlap, avg=True)
    #full_img = full_img.numpy()
   # full_img = np.moveaxis(full_img, 0, -1)

    #gt = unpatchify_predictions(gt.cpu(), (3, og_size[-2], og_size[-1]), patch_size=gset.patch_size, overlap=overlap, avg=True)
   # gt = gt.numpy()
   # gt = np.moveaxis(gt, 0, -1)
        #output = net(img)#.cpu()
       # output = output.cpu()
       # output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
      #  if net.n_classes > 1:
      #      mask = output.argmax(dim=1)
      #  else:
      #      mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy(), og_img, mask_gt #full_mask[0].long().squeeze().numpy(), gt


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default="C:\\Users\\Admin\\Desktop\\Gemsy\\External\\Pytorch-UNet\\checkpoints\\checkpoint_epoch56.pth", metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', default="1", help='Filenames of input images') # required=True
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true', default=True,
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.1,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'C:\\Users\\Admin\\Desktop\\Gemsy\\Data\\processed\\.out\\comparison\\{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 20, 21, 22]
    args.input = [f"{idx}" for idx in idxs]
    in_files = args.input
    out_files = get_output_filenames(args)

    features = ["full", "opacity", "dbscantuned"]
    n_channels = len(features) * 3
    #net = UNet(n_channels=n_channels, n_classes=args.classes, bilinear=args.bilinear)
    net = UNet_og(n_channels=n_channels, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        #img = Image.open(filename)

        mask, img, gt = predict_img(net=net,
                           full_img=filename,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device,
                           features=features)

        if not args.no_save:
            out_filename = out_files[i]
            mask_values = [0, 1]
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            #plot_img_and_mask_binary(img, mask, gt)
                        #plot_img_and_mask(img, mask, gt)
