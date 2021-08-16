# Acknowledgements
This code was adapted from a Pytorch implementation of the U-Net architecture made available by Alexandre Milesi. The original codebase is available at https://github.com/milesial/Pytorch-UNet/

# Usage
## train.py
The main training script used to train the models. Requires data/imgs and data/masks to be filled with training data Example usage: `python3 train.py -e 30 -s 1 -l 0.00001 -a ReversedUNet`

## highlight_plastics.py
Script to generate annotated output images with their counts prepended from a directory of input images. Uses the neural network as the output mask. Example usage: `python3 highlight_plastic_dir.py -m MODEL.pth -s 1.0 --input test --output out-imgs -a ReversedUNet`

## ground_truth.py
Script to generate annotated output images with their counts prepended from a directory of input images. Uses the ground truth masks of the images. Example usage: `python3 ground_truth.py -m MODEL.pth -s 1.0 --input test --masks test-masks --output out-imgs 
