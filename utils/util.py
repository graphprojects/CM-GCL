import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def evaluate_performance(labels, output):
    f1_micro = f1_score(labels.cpu().detach().numpy(),
                        output.max(1)[1].cpu().detach().numpy(), average='micro')
    f1_macro = f1_score(labels.cpu().detach().numpy(),
                        output.max(1)[1].cpu().detach().numpy(), average='macro')

    if labels.max() > 1:
        auc = roc_auc_score(labels.detach().cpu(),
                            F.softmax(output, dim=-1).detach().cpu(), average='macro',
                            multi_class='ovr')
    else:

        auc = roc_auc_score(labels.detach().cpu(),
                            F.softmax(output, dim=-1)[:, 1].detach().cpu(), average='macro')

        # auc = roc_auc_score(labels.detach().cpu(),
        #                     torch.nan_to_num(F.softmax(output, dim=-1)[:, 1], 1e-5).detach().cpu(), average='macro')

    return f1_micro, f1_macro, auc
