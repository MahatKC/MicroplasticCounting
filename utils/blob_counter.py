import os
import cv2
import numpy as np
from skimage.measure import label

project_base = '/home/ks/Projects/Microplastics'

image_dir = os.path.join(project_base, '/Data/segmentation_raw/imgs')
mask_dir = os.path.join(project_base, '/Data/segmentation_raw/masks')
out_dir = os.path.join(project_base, '/Data/test_out')


# # Read image
for mask_name in os.listdir(mask_dir):
    mask = os.path.join(mask_dir, mask_name)
    mask = cv2.imread(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    (thresh, mask) = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    con_map, num_blobs = label(mask, return_num=True, connectivity=None)

    new_mask_name = f'{num_blobs}_' + mask_name
    im_name = mask_name.replace('tiff', 'jpeg')
    im = cv2.imread(os.path.join(image_dir, im_name))
    new_im_name = f'{num_blobs}_' + im_name

    cv2.imwrite(os.path.join(out_dir, new_mask_name), mask)
    cv2.imwrite(os.path.join(out_dir, new_im_name), im)
