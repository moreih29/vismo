
from typing import Union

import torch


def empty_cache(device: Union[str, torch.device]):
    if is_cuda(device):
        with torch.cuda.device(device):
            empty_cache()
    elif is_mps(device):
        torch.mps.empty_cache()
        

def is_cpu(device: Union[str, torch.device]):
    device = torch.device(device)
    return device.type == 'cpu'


def is_cuda(device: Union[str, torch.device]):
    device = torch.device(device)
    return device.type == 'cuda' and torch.has_cuda


def is_mps(device: Union[str, torch.device]):
    device = torch.device(device)
    return device.type == 'mps' and torch.has_mps