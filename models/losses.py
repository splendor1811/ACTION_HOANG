import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntrophyLoss(nn.Module):
    def __init__(self, args, reweighted=None, reweighted_idx=None):
        super(CrossEntrophyLoss, self).__init__()
        self.args = args
        self.reweighted = reweighted
        self.reweighted_idx = reweighted_idx

    def forward(self, output, target):
        if self.reweighted:
            loss = nn.CrossEntropyLoss(weight=torch.FloatTensor(self.reweighted_idx)).cuda(self.args['device'])(output, target)
        else:
            loss = nn.CrossEntropyLoss().cuda(self.args['device'])(output, target)
        return loss


class binary_cross_entrophy_with_logits(nn.Module):
    def __init__(self, ratio = None):
        super(binary_cross_entrophy_with_logits, self).__init__()
        self.ratio = ratio

    @staticmethod
    def ratio2weight(targets, ratio, option="Loss3"):
        ratio = torch.from_numpy(ratio).type_as(targets)
        # print(ratio)
        # print(targets)
        if option == "Loss1":
            # --------------------- Dangwei li TIP20 ---------
            pos_weights = targets * (1 - ratio)
            neg_weights = (1 - targets) * ratio
            weights = torch.exp(neg_weights + pos_weights)
        elif option == "Loss2":
            # --------------------- AAAI ---------------------
            pos_weights = torch.sqrt(1 / (2 * ratio.sqrt())) * targets
            neg_weights = torch.sqrt(1 / (2 * (1 - ratio.sqrt()))) * (1 - targets)
            weights = pos_weights + neg_weights
        else:
            # --------------------- CVPR2021 -----------------
            pos_weights = ((1 / ratio) / ((1 / ratio) + (1 / (1 - ratio)))) * targets
            neg_weights = ((1 / (1 - ratio)) / ((1 / ratio) + (1 / (1 - ratio)))) * (1 - targets)
            weights = pos_weights + neg_weights

        # for RAP dataloader, targets element may be 2, with or without smooth, some element must greater than 1
        weights[targets > 1] = 0.0
        return weights

    def forward(self, output, target):
        if self.ratio is not None:
            sample_weight = self.ratio2weight(target, ratio=self.ratio)
            loss = F.binary_cross_entropy_with_logits(output, target, weight=sample_weight, reduction='none')
        else:
            loss = F.binary_cross_entropy_with_logits(output, target, reduction='none')
        loss = loss.sum(1).mean()
        return loss
