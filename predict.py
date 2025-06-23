import argparse
import logging
import os
from pathlib import Path

from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torchvision

from unet.unet_model_og import UNet_og
from utils.data_loading import GemsyDataset
from unet import UNet
from utils.utils import plot_img_and_mask, plot_img_and_mask_binary

dir_img = Path("C:\\Users\\Admin\\Desktop\\Gemsy\\Data\\processed\\.in\\features\\")
dir_img = Path("C:\\Users\\Admin\\Desktop\\Gemsy\\Data\\processed\\.pred\\features\\")
dir_mask = Path("C:\\Users\\Admin\\Desktop\\Gemsy\\Data\\processed\\.in\\masks\\")
dir_mask = Path("C:\\Users\\Admin\\Desktop\\Gemsy\\Data\\processed\\.pred\\masks\\")
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
                overlap = 0,
                target_size = None):
    net.eval()
    gset = GemsyDataset(dir_img, dir_mask, patch_size, patches_per_image, img_scale, features=features, target_size=target_size)
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
           # out = torchvision.transforms.functional.rgb_to_grayscale(batch) #net(batch)
            out = net(batch)
            #output = F.interpolate(out, (full_img.size[1], full_img.size[0]), mode='bilinear').cpu()
            #if net.n_classes > 1:
            #    mask = out.argmax(dim=1)
            #else:
            #    mask = torch.sigmoid(out) > out_threshold

           # outputs.append(mask.cpu())
            outputs.append(torch.sigmoid(out).cpu())

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

    return mask[0].squeeze().numpy(), og_img, mask_gt
   # return mask[0].long().squeeze().numpy(), og_img, mask_gt #full_mask[0].long().squeeze().numpy(), gt


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default="", metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', default="1", help='Filenames of input images') # required=True
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true', default=True,
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
 #   parser.add_argument('--mask-threshold', '-t', type=float, default=0.50,
  #                      help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(args, runname):
    def _generate_name(fn):
        path = f'C:\\Users\\Admin\\Desktop\\Gemsy\\Data\\processed\\.out\\val\\final\\{runname}'
        Path(path).mkdir(parents=True, exist_ok=True)

        return f'{path}\\{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
      #  out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.float32)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return out
    #return Image.fromarray(out)


run_ids={
    "f696db5o":"quiet-grass-232",
    "2jhd7j36":"fanciful-armadillo-234",
    "w1v5yru4":"peach-sound-238",
    "zyqug84w":"hirogen-farpoint-260",
    "9sor4qn4":"nemesis-maquis-262",
    "tshedsbx":"andorian-sicko-263",
    "feebsgsq":"floral-moon-282",
    "zwznszk1":"colorful-forest-284",
    "e58564qs":"desert-sunset-285",
    "2tdnjrx9": "robust-shape-286",
    "jlkdtn4q": "cerulean-glade-289",
    "04xdnf4d": "apricot-fog-290",
    "7cjp6k59": "chocolate-blaze-291",
    "0vqyi0s2": "polar-breeze-292",
    "iwk1bsaf": "clean-dream-293",
    "k3gqimow": "olive-planet-303",
    "c81vfcbd": "twilight-glade-307",
    "08054sot": "driven-voice-309"
}

# 70 epoch
runs={
    "quiet-grass-232": {"type":"Custom", "features":["full","opacity","dbscan"]},
    "fanciful-armadillo-234": {"type":"OG", "features":["full","opacity","dbscan"]},
    "peach-sound-238": {"type":"Custom", "features":["opacity","full","dbscan"]},
   # "hirogen-farpoint-260": {"type":"Custom", "features":["opacity","dbscantuned","full"]},
  #  "nemesis-maquis-262": {"type":"OG", "features":["full","opacity","dbscantuned"]},
    "andorian-sicko-263": {"type":"OG", "features":["full","opacity","dbscantuned"]}, # ** 60 epoch
    "floral-moon-282": {"type":"OG", "features":["full","opacity","dbscantuned"]},
  ##  "colorful-forest-284": {"type":"OG", "features":["dbscantuned","full","opacity"]},
    "desert-sunset-285": {"type":"OG", "features":["full","dbscantuned","opacity"]}, # ** 45 epoch
    "robust-shape-286": {"type":"Custom", "features":["full","dbscantuned","opacity"]},
    "cerulean-glade-289": {"type":"Custom", "features":["full","dbscantuned","opacity"]}, #** 60 epoch
    "apricot-fog-290": {"type":"Custom", "features":["full","dbscantuned","opacity"]}, # ** 55 epoch
    "chocolate-blaze-291": {"type":"Custom", "features":["full","dbscantuned","opacity"]},
    "polar-breeze-292": {"type":"Custom", "features":["gt"]},
    "clean-dream-293": {"type":"Custom", "features":["gt"]},
}


# Fix failed runs
# 45
#runs = {
#    "desert-sunset-285": {"type":"OG", "features":["full","dbscantuned","opacity"]},  
#}

# 55
runs = {
    "apricot-fog-290": {"type":"Custom", "features":["full","dbscantuned","opacity"]},  # 55 epoch
}

# 60 epoch
runs = {
    "andorian-sicko-263": {"type":"OG", "features":["full","opacity","dbscantuned"]},  # 60
}

# Additional runs to showcase best-case performance
# 170 eepoch
runs={
    "polar-breeze-292": {"type":"Custom", "features":["gt"]},
    "clean-dream-293": {"type":"Custom", "features":["gt"]},
}
# 100 epoch
runs={
    "quiet-grass-232": {"type":"Custom", "features":["full","opacity","dbscan"]},
    "fanciful-armadillo-234": {"type":"OG", "features":["full","opacity","dbscan"]},
    "robust-shape-286": {"type":"Custom", "features":["full","dbscantuned","opacity"]},
    "chocolate-blaze-291": {"type":"Custom", "features":["full","dbscantuned","opacity"]},
}

# 60 max?
runs={
    "cerulean-glade-289": {"type":"Custom", "features":["full","dbscantuned","opacity"]},
}

runs={
    "olive-planet-303": {"type":"Custom", "features":["gt"]},
}

runs={
    "twilight-glade-307": {"type":"Custom", "features":["gt"]},
}

runs={
    "driven-voice-309": {"type":"Custom", "features":["gt"]},
}

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    model_path = "C:\\Users\\Admin\\Desktop\\Gemsy\\External\\Pytorch-UNet\\checkpoints"
    ckpt = 180 #60
    target_size = (6000, 4000) #(4500, 3000)

    mask_threshold = None

    for run_name, desc in runs.items():
        try:
            # CHANGE THESE ACCORDING TO RUN-SPECIFIC PARAMETERS
           # runname = "apricot-fog-290" 

            model_type = desc["type"]
            features = desc["features"] #["full", "dbscantuned", "opacity"]#, "dbscantuned", "sized"]



            full_model = f"{model_path}\\{run_name}\\checkpoint_epoch{ckpt}.pth"
       #     idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 20, 21, 22]
          #  idxs = [8, 10, 15, 20]
            idxs = ['1',
                '10',
                '11',
                '12',
                '14',
                '15',
                '2',
                '20',
                '21',
                '22',
                '3',
                '4',
                '5',
                '6',
                '7',
                '8',
                '9',
                'e-10',
                'e-11',
                'e-12',
                'e-13',
                'e-14',
                'e-16',
                'e-3',
                'e-6',
                'e-8',
                'e-9']
            idxs= ['e-10', 'e-11', 'e-12', 'e-13', 'e-14', 'e-15', 'e-16', 'e-17', 'e-18', 'e-19', 'e-20', 'e-21', 'e-22', 'e-23', 'e-24', 'e-25', 'e-26', 'e-27', 'e-28', 'e-29', 'e-3', 'e-30', 'e-31', 'e-32', 'e-33', 'e-34', 'e-35', 'e-36', 'e-37', 'e-38', 'e-39', 'e-4', 'e-40', 'e-5', 'e-6', 'e-7', 'e-8', 'e-9']
            args.input = [f"{idx}" for idx in idxs]
            in_files = args.input
            out_files = get_output_filenames(args, f"{run_name}_{ckpt}")



            n_channels = len(features) * 3
            if model_type == "OG":
                net = UNet_og(n_channels=n_channels, n_classes=args.classes, bilinear=args.bilinear)
            else:
                net = UNet(n_channels=n_channels, n_classes=args.classes, bilinear=args.bilinear)


            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logging.info(f'Loading model {full_model}')
            logging.info(f'Using device {device}')

            net.to(device=device)
            state_dict = torch.load(full_model, map_location=device)
            mask_values = state_dict.pop('mask_values', [0, 1])
            net.load_state_dict(state_dict)

            logging.info('Model loaded!')

            for i, filename in enumerate(in_files):
                logging.info(f'Predicting image {filename} ...')
                #img = Image.open(filename)

                mask, img, gt = predict_img(net=net,
                                   full_img=filename,
                                   scale_factor=args.scale,
                                   out_threshold=mask_threshold,#args.mask_threshold,
                                   device=device,
                                   features=features,
                                   target_size=target_size)

                if not args.no_save:
                    out_filename = out_files[i]
                    mask_values = [0, 1]
                    #result = mask_to_image(mask, mask_values)
                   # plt.imshow(result*255, cmap='viridis')

                   # plt.colorbar()
                   # plt.savefig(out_filename)
                    #result = 
                  #  result.save(out_filename)

                    # Apply colormap (returns RGBA image)
                    colored_array = matplotlib.cm.inferno(mask)  # shape: (H, W, 4), float32 in [0, 1]

                    # Convert to 8-bit per channel RGB image (drop alpha)
                    colored_array = (colored_array[:, :, :3] * 255).astype(np.uint8)

                    # Convert back to PIL Image
                    colored_img = Image.fromarray(colored_array)

                    # Save the result
                    colored_img.save(out_filename)


                    logging.info(f'Mask saved to {out_filename}')

                if args.viz:
                    logging.info(f'Visualizing results for image {filename}, close to continue...')
                    #plot_img_and_mask_binary(img, mask, gt)
                                #plot_img_and_mask(img, mask, gt)

        except Exception as e:
            print(e)
            print(run_name)
