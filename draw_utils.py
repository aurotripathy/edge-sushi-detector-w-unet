import torch
import cv2
import matplotlib.pyplot as plt


def prep_img_for_inference(device, sample, scale_factor=0.5):
    img = torch.from_numpy(cv2.resize(
        sample, None, fx=scale_factor, fy=scale_factor)).to(device)
    # convert to CHW from HWC
    img = img.permute(2, 0, 1)
    # convert to NCHW
    img = img.unsqueeze(0)
    # convert from the OpenCV byte representation
    img = img.type(torch.FloatTensor)
    # scale to [0..1]
    img /= 255
    return img


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
    plt.figure(figsize=(20, 20))
    # plt.subplot(1, 2, 1)
    plt.imshow(sample, 'gray', interpolation='none')
    # plt.subplot(1, 2, 2)
    # plt.imshow(sample, 'gray', interpolation='none')
    plt.imshow(mask, 'jet', interpolation='none', alpha=0.7)
    plt.show()


def post_process(output, threshold=0.5, has_probs=False):
    if not has_probs:
        probs = torch.sigmoid(output)
    else:
        probs = output
    probs = probs.squeeze(0)

    out_mask = (probs > threshold).cpu().numpy().astype("int") * 255
    return out_mask
