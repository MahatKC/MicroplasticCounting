import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    tot_precision = 0
    tot_recall = 0
    count_precision = 0
    count_recall = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()

                true_pos = ((pred == 1) & (true_masks == 1)).sum().item()
                false_pos = ((pred == 1) & (true_masks == 0)).sum().item()
                false_neg = ((pred == 0) & (true_masks == 1)).sum().item()

                if true_pos != 0 and false_pos != 0:
                    precision = true_pos / (true_pos + false_pos)
                    tot_precision += precision
                    count_precision += 1

                if true_pos != 0 and false_neg != 0:
                    recall = true_pos / (true_pos + false_neg)
                    tot_recall += recall
                    count_recall += 1



            pbar.update()

    net.train()
    if count_recall == 0:
        count_recall = 1
    if count_precision == 0:
        count_precision = 1
    return (tot / n_val, tot_precision / count_precision, tot_recall / count_recall)
