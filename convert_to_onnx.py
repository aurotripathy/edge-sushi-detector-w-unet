import onnxruntime
import onnx
import torch
import torch.nn as nn
import unet
import cv2
from pudb import set_trace
import numpy as np
import matplotlib.pyplot as plt
from draw_utils import prep_img_for_inference, draw_image_superimposed_w_mask, post_process
import os

# Define input/output locations
root_outputs_path = './outputs'

# forcibly naming 'bananas' to be download compatible
model_path = os.path.join(root_outputs_path, "model")
model_file = os.path.join(model_path, "bananas.pth")

onnx_path = os.path.join(root_outputs_path, "onnx")
onnx_output = os.path.join(onnx_path, "bananas.onnx")


device = torch.device('cpu')
net = unet.UNet(n_channels=3, n_classes=1, bilinear=False)
net.to(device=device)
net.load_state_dict(torch.load(model_file, map_location=device))

# The original code did not have output activation. Bake it in.
model = nn.Sequential(net, nn.Sigmoid())

# Prep the input dimensions

inference_file_name = "IMG_1082-size_818_616.JPG"
root_data_path = '.'
sample_path = os.path.join(
    root_data_path, 'test-batch-1_size_818_616', inference_file_name)
sample_img = cv2.imread(sample_path)
dummy_input = prep_img_for_inference(device, sample_img)
print(
    f"These will be the fixed dimensions of any incoming images: {dummy_input.shape}")


# Do the conversion with the right opset_version

# opset_version = 11 to support up-convolutional layers
torch.onnx.export(model, dummy_input, onnx_output, opset_version=11,
                  export_params=True, input_names=["input"],
                  output_names=["output"], verbose=False)

# 7. Check with ONNX Runtime

# We are going to use a CPU-based version of ONNX runtime in order
# to avoid CUDA compat problems
# %pip install onnxruntime

# Check the model


onnx_model = onnx.load(onnx_output)
onnx.checker.check_model(onnx_model)
print('The model is checked!')

# Perform inference with the model


ort_session = onnxruntime.InferenceSession(onnx_output)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# compute ONNX Runtime output prediction
sample_input = prep_img_for_inference(device, sample_img)
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(sample_input)}
ort_outs = ort_session.run(None, ort_inputs)

threshold = 0.5
scale_factor = 0.5
out_mask = post_process(torch.from_numpy(
    ort_outs[0]), threshold=threshold, has_probs=True)
out_mask = np.squeeze(out_mask, 0)
sample_for_display = cv2.resize(
    sample_img, None, fx=scale_factor, fy=scale_factor)
draw_image_superimposed_w_mask(sample_for_display, out_mask)
