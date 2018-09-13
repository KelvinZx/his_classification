import torch
import torch.nn

def load_pretrain_state_dict(own_dict, state_dict):
    for name, param in state_dict.items():
        if name not in own_dict:
            continue
        if isinstance(param, Parameter):
            param = param.data
        own_dict[name].copy_(param)
