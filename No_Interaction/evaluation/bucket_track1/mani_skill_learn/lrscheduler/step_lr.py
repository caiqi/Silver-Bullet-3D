import torch
from .builder import LR_SCHEDULER

@LR_SCHEDULER.register_module()
class StepLR(torch.optim.lr_scheduler.StepLR):
    def __init__(
        self, 
        optimizer, 
        step_size, 
        gamma=0.8
    ):
        super(StepLR, self).__init__(
            optimizer, 
            step_size, 
            gamma
        )