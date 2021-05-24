#  Draw a sample image and mask

from PIL import Image
import numpy as np
import os
import cv2
import matplotlib.pylab as plt
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--mask', required=True,
                    type=str, help='the path to mask')
parser.add_argument('--image', required=True,
                    type=str, help='the path to image')

args = parser.parse_args()
img_file = args.image
mask_file = args.mask

# root_data_path = './data'

# train_path = os.path.join(root_data_path, 'images')
# mask_path = os.path.join(root_data_path, 'masks')
# file_name = "2021-01-11T224237.988294Z"
# img_file = os.path.join(train_path, f"{file_name}.jpg")
# mask_file = os.path.join(mask_path, f"{file_name}.png")


os.path.exists(img_file),
os.path.exists(mask_file)

sample = cv2.imread(img_file)

# OpenCV doesn't read .gif files. Workaround
mask_pil = Image.open(mask_file)
mask = np.array(mask_pil)


def draw_image_mask(sample, mask):
    fig, ax = plt.subplots(1, 2, figsize=(20, 20))
    ax[0].imshow(cv2.cvtColor(sample, cv2.COLOR_BGR2RGB))
    ax[0].axis('off')
    ax[0].title.set_text('Sample Image')
    ax[1].imshow(mask)
    ax[1].axis('off')
    ax[1].title.set_text('Sample Mask')
    plt.show()


def draw_image_superimposed_w_mask(sample, mask):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(sample, 'gray', interpolation='none')
    plt.subplot(1, 2, 2)
    plt.imshow(sample, 'gray', interpolation='none')
    plt.imshow(mask, 'jet', interpolation='none', alpha=0.7)
    plt.show()


# draw_image_mask(sample, mask)
draw_image_superimposed_w_mask(sample, mask)
