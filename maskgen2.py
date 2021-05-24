import os
import cv2
import json
import numpy as np
import shutil
from pudb import set_trace

source_folder = os.path.join(os.getcwd(), 'resized_images')
dest_folder = os.path.join(os.getcwd(), 'resized_masks')
shutil.rmtree(dest_folder, ignore_errors=True)
os.mkdir(dest_folder)


print('The source folder', source_folder)
print('The dest folder', dest_folder)

json_path = "via_project_19May2021_21h41m_json(1).json"
MASK_WIDTH, MASK_HEIGHT = 614, 818
print('masks will be of size', MASK_WIDTH, MASK_HEIGHT)

# Read JSON VIA file
with open(json_path) as f:
    via_data = json.load(f)


for itr in via_data:
    print('processing file', itr)
    mask = np.zeros((MASK_WIDTH, MASK_HEIGHT))
    for count in range(len(via_data[itr]['regions'])):
        x_points = via_data[itr]["regions"][count]["shape_attributes"]["all_points_x"]
        y_points = via_data[itr]["regions"][count]["shape_attributes"]["all_points_y"]
        all_points = []
        for x, y in zip(x_points, y_points):
            all_points.append((x, y))
        arr = np.array(all_points)
        cv2.fillPoly(mask, [arr], color=(255))
    cv2.imwrite(os.path.join(dest_folder, itr.split('.')[0]) + '.png', mask)
