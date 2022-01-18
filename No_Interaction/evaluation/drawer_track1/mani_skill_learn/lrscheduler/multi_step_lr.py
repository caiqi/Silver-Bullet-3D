import torch
from .builder import LR_SCHEDULER

@LR_SCHEDULER.register_module()
class MultiStepLR(torch.optim.lr_scheduler.MultiStepLR):
    def __init__(
        self, 
        optimizer, 
        steps, 
        gamma=0.1,
        last_epoch=-1,
    ):
        super(MultiStepLR, self).__init__(
            optimizer, 
            steps, 
            gamma,
            last_epoch
        )