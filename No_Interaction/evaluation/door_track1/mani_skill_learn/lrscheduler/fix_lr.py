import torch
from torch.optim.lr_scheduler import LambdaLR
from .builder import LR_SCHEDULER

@LR_SCHEDULER.register_module()
class FixLR(LambdaLR):

    def __init__(
        self, 
        optimizer, 
        last_epoch=-1
    ):
        super(FixLR, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        return 1.