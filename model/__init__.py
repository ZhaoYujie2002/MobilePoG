import torch
from .affnet import AFFNetModel
from .itracker import ITrackerModel

def init_model(model_name, ckpt_path=None):
    if model_name == 'iTracker':
        model = ITrackerModel()
    elif model_name == 'AFFNet':
        model = AFFNetModel()
    else:
        raise RuntimeError("config model name error")
    if ckpt_path is not None: 
        model.load_state_dict(torch.load(ckpt_path))

    return model