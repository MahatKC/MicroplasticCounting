import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
import imutils
import statistics
import argparse
import logging
import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from unet import UNet
from reversed_unet import ReversedUNet
from semi_dense_unet import SemiDenseUNet

from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
import image_slicer
from image_slicer import join


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    data_transforms = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [ 0.6976, 0.7220, 0.7588],
                                                     std = [ 0.1394, 0.1779, 0.2141 ])])
    img = BasicDataset.preprocess(full_img, scale_factor, 'img')
    img = data_transforms(img)

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i',
                        help='input diretory with images', required=True)

    parser.add_argument('--output', '-o',
                        help='output directory')
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1.0)
    parser.add_argument('-a', '--architecture', dest='architecture', type=str, default="UNet",
                        help='Architecture, choose from UNet, ReversedUNet and SemiDenseUNet')

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    in_dir = args.input
    out_dir = args.output

    if args.architecture == "UNet":
        net = UNet(n_channels=3, n_classes=1, bilinear=False)
    elif args.architecture == "ReversedUNet":
        net = ReversedUNet(n_channels=3, n_classes=1, bilinear=False)
    elif args.architecture == "SemiDenseUNet":
        net = SemiDenseUNet(n_channels=3, n_classes=1, bilinear=False)
    else:
        logging.info(f"Architecture {args.architecture} not found")
        exit(0)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")
    for i, fn in enumerate(os.listdir(in_dir)):
        logging.info("\nPredicting image {} ...".format(fn))

        full_img = Image.open(os.path.join(in_dir, fn))
        tiles = image_slicer.slice(os.path.join(in_dir, fn), number_tiles=4, row=2)
        masks = []

        for tile in tiles:
            img = Image.open(tile.filename)
            mask = predict_img(net=net,
                               full_img=img,
                               scale_factor=args.scale,
                               out_threshold=args.mask_threshold,
                               device=device)

            mask = np.where(mask > 0, 255, 0)
            mask = cv2.convertScaleAbs(mask)
            masks.append(mask)
            os.remove(tile.filename)

        mask_1 = np.hstack((masks[0], masks[1]))
        mask_2 = np.hstack((masks[2], masks[3]))
        mask = np.vstack((mask_1, mask_2))

        kernel_dilate = np.ones((15,15),np.uint8)
        kernel_erode = np.ones((9,9),np.uint8)
        dilation = cv2.dilate(mask ,kernel_dilate, iterations = 1)
        mask = cv2.erode(dilation, kernel_erode, iterations = 1)

        contours = cv2.findContours(mask,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if (len(contours[0]) != 0):

            cnts = imutils.grab_contours(contours)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

            rect_areas = []
            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                rect_areas.append(w * h)

            avg_area = statistics.mean(rect_areas)

            transparency = 1
            full_img = np.array(full_img, dtype=np.float)
            full_img /= 255.0

            mask = cv2.merge((mask, mask, mask))
            mask = np.where(mask > 0, np.array([1, 0, 0]), np.array([0, 0 ,0]))
            mask = mask * transparency
            full_img = mask + full_img * (1.0 - mask)

            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(full_img,(x,y),(x+w,y+h),(255,0,0),2)
                # if cnt_area < 0.1 * avg_area:
                #     mask[y:y + h, x:x + w] = 0

        if (type(full_img) is np.ndarray):
            full_img = mask_to_image(full_img)
        full_img.save(os.path.join(out_dir, f'c{len(contours[0])}_' + fn))

        logging.info("Mask saved to {}".format(os.path.join(out_dir, fn)))
