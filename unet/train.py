import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from eval import eval_net
from unet import UNet
from reversed_unet import ReversedUNet
from semi_dense_unet import SemiDenseUNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

dir_img = 'data/imgs/'
dir_mask = 'data/masks/'
dir_checkpoint = 'checkpoints/'

torch.set_printoptions(edgeitems=1000)


def train_net(net,
              device,
              epochs=5,
              batch_size=2,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=1.0):

    data_transforms = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean = [ 0.6976, 0.7220, 0.7588 ],
                            std = [ 0.1394, 0.1779, 0.2141 ])])


    dataset = BasicDataset(dir_img, dir_mask, img_scale, img_transforms=data_transforms)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)


    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    #scaler = torch.cuda.amp.GradScaler()
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=4, gamma=0.8)

    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                #   print(true_masks)
                #   print(true_masks.max())
                #   print(true_masks.min())

                # with torch.cuda.amp.autocast():
                #     masks_pred = net(imgs)
                #     loss = criterion(masks_pred, true_masks)
                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                #scaler.scale(loss).backward()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                #scaler.update()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score, precision, recall = eval_net(net, val_loader, device)
                    #scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)
                    logging.info('Precision: {}'.format(precision))
                    logging.info('Recall: {}'.format(recall))
                    writer.add_scalar('Precision', precision, global_step)
                    writer.add_scalar('Recall', recall, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
        scheduler.step()

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1.0,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-n', '--normalisation', action='store_true',
                        help='Print the mean and standard deviation of dataset and exit')
    parser.add_argument('-a', '--architecture', dest='architecture', type=str, default="UNet",
                        help='Architecture, choose from UNet, ReversedUNet and SemiDenseUNet')

    return parser.parse_args()

def get_mean_std(loader):
    channels_sum = [0] * 3
    channels_squared_sum = [0] * 3
    num_batches = 0

    mean = [0] * 3
    std = [0] * 3

    for b in loader:
        channels_sum[0] += b['image'][0][0].mean()
        channels_sum[1] += b['image'][0][1].mean()
        channels_sum[2] += b['image'][0][2].mean()

        channels_squared_sum[0] += (b['image'][0][0] ** 2).mean()
        channels_squared_sum[1] += (b['image'][0][1] ** 2).mean()
        channels_squared_sum[2] += (b['image'][0][2] ** 2).mean()

        num_batches += 1

    for i, _ in enumerate(mean):
        mean[i] = channels_sum[i] / num_batches
    for i, _ in enumerate(std):
        std[i] = (channels_squared_sum[i] / num_batches - mean[i] ** 2) ** 0.5

    return mean, std

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N

    if args.normalisation:
        data_transforms = transforms.Compose([
                                transforms.ToTensor()])
        dataset = BasicDataset(dir_img, dir_mask, args.scale, img_transforms=data_transforms)
        dataset_loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True, num_workers=8, pin_memory=True)

        mean, std = get_mean_std(dataset_loader)
        logging.info(f"Mean: {mean}")
        logging.info(f"Std: {std}")

        exit(0)

    if args.architecture == "UNet":
        net = UNet(n_channels=3, n_classes=1, bilinear=False)
    elif args.architecture == "ReversedUNet":
        net = ReversedUNet(n_channels=3, n_classes=1, bilinear=False)
    elif args.architecture == "SemiDenseUNet":
        net = SemiDenseUNet(n_channels=3, n_classes=1, bilinear=False)
    else:
        logging.info(f"Architecture {args.architecture} not found")
        exit(0)

    logging.info(f'Network:\n'
                 f'\t{args.architecture} architecture used\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    torch.save(net.state_dict(), 'MODEL.pth')
