from ..utils.meta import Registry, build_from_cfg

LR_SCHEDULER = Registry('lr_scheduler')

def build_lr_scheduler(cfg, default_args=None):
    return build_from_cfg(cfg, LR_SCHEDULER, default_args)