
from typing import Callable, Dict
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils import is_cuda, StopWatch


class LeNet5(nn.Module):
    def __init__(self,
                 nc: int = 10,
                 ) -> None:
        super().__init__()
        
        self.c1 = nn.Conv2d(in_channels=1,
                            out_channels=6,
                            kernel_size=(5, 5))
        self.s2 = nn.AvgPool2d(kernel_size=2,
                               stride=2)
        self.c3 = nn.Conv2d(in_channels=6,
                            out_channels=16,
                            kernel_size=(5, 5))
        self.s4 = nn.AvgPool2d(kernel_size=2,
                               stride=2)
        self.c5 = nn.Conv2d(in_channels=16,
                            out_channels=120,
                            kernel_size=(5, 5))
        self.f6 = nn.Linear(in_features=120,
                            out_features=84)
        self.out = nn.Linear(in_features=84,
                             out_features=nc)
        
    def forward(self, 
                x: torch.Tensor,
                ) -> torch.Tensor:
        x = self.forward_feature(x)
        x = self.forward_head(x)
        
        return x
    
    def forward_feature(self, 
                        x: torch.Tensor,
                        ) -> torch.Tensor:
        x = self.c1(x)
        x = self.s2(x)
        x = torch.sigmoid_(x)
        x = self.c3(x)
        x = self.s4(x)
        x = torch.sigmoid_(x)
        x = self.c5(x)
        
        return x
    
    def forward_head(self,
                     x: torch.Tensor,
                     amp: int = 1.7159,
                     ) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        x = self.f6(x)
        x = amp * torch.tanh_(x)
        x = self.out(x)
        
        return x
    
    def fit(self,
            train_loader: DataLoader,
            criterion: nn.Module,
            optimizer: optim.Optimizer,
            epochs: int,
            metrics: Dict = {},
            valid_loader: DataLoader = None,
            fp16: bool = False,
            on_after_epoch: Callable = None):
        
        scaler = torch.cuda.amp.GradScaler(enabled=fp16 and is_cuda(self.device))
        nt = len(train_loader)
        nv = len(valid_loader) if valid_loader else -1
        history = defaultdict(list)
        for epoch in range(epochs):
            with StopWatch(on_end=on_after_epoch,
                           args=(history,)):
                self.train()
                tl = 0
                for inp, targ in tqdm(train_loader, desc=f'train {epoch}'):
                    inp = inp.to(self.device)
                    targ = targ.to(self.device)
                    with torch.cuda.amp.autocast(enabled=fp16):
                        output = self.forward(inp)
                        train_loss = criterion(output, targ)
                        output = output.argmax(dim=-1)
                        optimizer.zero_grad()
                        scaler.scale(train_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        tl += train_loss.item()
                        for m in metrics.values():
                            m.update(targ, output)
                tl /= nt
                history['train_loss'].append(tl)
                for k, m in metrics.items():
                    history[f'train_{k}'].append(m.compute())
                
                if valid_loader:
                    self.eval()
                    for m in metrics.values():
                        m.reset()
                    vl = 0
                    for inp, targ in tqdm(valid_loader, desc=f'valid {epoch}'):
                        inp = inp.to(self.device)
                        targ = targ.to(self.device)
                        output = self.forward(inp)
                        valid_loss = criterion(output, targ)
                        output = output.argmax(dim=-1)
                        vl += valid_loss.item()
                        for m in metrics.values():
                            m.update(targ, output)
                    vl /= nv
                    history['valid_loss'].append(tl)
                    for k, m in metrics.items():
                        history[f'valid_{k}'].append(m.compute())
    
    @torch.no_grad()
    def predict(self,
                x: np.ndarray,
                ) -> torch.Tensor:
        self.eval()
        if len(x.shape) == 3:
            x = x[None]
        
        x = (torch.as_tensor(x, device=self.device)
             .permute(0, 3, 1, 2)
             .float()
             / 255)
        
        x = F.interpolate(x, (32, 32))
        return self.predict_torch(x)
        
    def predict_torch(self,
                      x: torch.Tensor,
                      ) -> torch.Tensor:
        return self.forward(x)
    
    @property
    def device(self):
        return self.c1.weight.device
    