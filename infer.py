import torch
import os
from dice_loss import dice
import unet
import cv2
from pudb import set_trace
import numpy as np
import matplotlib.pyplot as plt
from draw_utils import prep_img_for_inference, draw_image_superimposed_w_mask, post_process

import argparse

parser = argparse.ArgumentParser(description='Provide an image for interence',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-i', '--image-file', type=str, required=True,
                    help='Path to image')
args = parser.parse_args()

set_trace()
# Infer using the original model
# This simply validates that the model works

# Prep the network

model_file = os.path.join('./outputs', "checkpoints/model.pth")
device = torch.device('cpu')
net = unet.UNet(n_channels=3, n_classes=1, bilinear=False)
net.to(device=device)
net.load_state_dict(torch.load(model_file, map_location=device))

# Prep the image

# We scale the image by this factor during training
scale_factor = 0.5


sample_path = args.image_file

# set_trace()
sample = cv2.imread(sample_path)
img = prep_img_for_inference(device, sample, scale_factor=scale_factor)

# Pipe it through

threshold = 0.5


net.eval()
with torch.no_grad():
    output = net(img)
    out_mask = post_process(output, threshold=threshold)


out_mask = np.squeeze(out_mask, 0)
sample_for_display = cv2.resize(sample, None, fx=scale_factor, fy=scale_factor)

draw_image_superimposed_w_mask(cv2.cvtColor(
    sample_for_display, cv2.COLOR_BGR2RGB), out_mask)

# draw_image_superimposed_w_mask(
#     cv2.cvtColor(sample, cv2.COLOR_BGR2RGB), out_mask)


# How accurate?
# Compute Dice Coefficient


# cv2.resize expects (width, height)
dice(cv2.resize(out_mask, (out_mask.shape[1], out_mask.shape[0])), out_mask)
