from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
# from optimizers import Optimizer
class Schedulers():
    def __init__(self, cfgs, optimizer ):
        self.cfgs = cfgs
        self.optimizer = optimizer

    def adjust_learning_rate(self, epoch):
        if epoch < self.cfgs['warm_up_epoch']:
            lr = self.cfgs['optimizer']['lr'] * (epoch + 1) / self.cfgs['warm_up_epoch']
        else:
            lr = self.cfgs['optimizer']['lr'] * (
                0.1 ** np.sum(epoch >= np.array(self.cfgs['step'])))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def get_schedulers(self, epoch=None):
        if epoch is not None:
            scheduler = self.adjust_learning_rate(epoch=epoch)
        else:
            scheduler = CosineAnnealingLR(self.optimizer, self.cfgs['num_epoch'], eta_min=self.cfgs['lr_config']['min_lr'])
        return scheduler