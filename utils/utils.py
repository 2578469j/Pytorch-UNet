import matplotlib.pyplot as plt


def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()

def plot_img_and_mask_binary(img, mask, gt):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[1].set_title('GT')
    ax[1].imshow(gt)
    ax[2].set_title(f'Mask')
    ax[2].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()
