
from typing import Callable, Dict, List
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils import is_cuda, StopWatch


class BaseBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self,
                x: torch.Tensor,
                ) -> torch.Tensor:
        x = self.forward_feature(x)
        x = self.forward_head(x)
        return x
    
    def forward_feature(self,
                        x: torch.Tensor,
                        ) -> torch.Tensor:
        raise NotImplementedError
    
    def forward_head(self,
                     x: torch.Tensor,
                     ) -> torch.Tensor:
        raise NotImplementedError
    
    def fit(self,
            train_loader: DataLoader,
            criterion: nn.Module,
            optimizer: optim.Optimizer,
            epochs: int,
            metrics: Dict = {},
            valid_loader: DataLoader = None,
            fp16: bool = False,
            on_after_epoch: Callable = None,
            multi_gpus: List[int] = []):
        
        for device in multi_gpus:
            if device < 0 or device >= torch.cuda.device_count():
                raise IndexError(f'Invalid GPU id. cuda:{device} is not available.')
        
        if len(multi_gpus):
            current_device = self.device
            eval_func = self.evaluate
            self = nn.DataParallel(self, device_ids=multi_gpus)
            self.device = current_device
            self.evaluate = eval_func
        
        fp16 = fp16 and is_cuda(self.device)
        scaler = torch.cuda.amp.GradScaler(enabled=fp16)
        history = defaultdict(list)
        for epoch in range(epochs):
            with StopWatch(on_end=on_after_epoch,
                           args=(history,)):
                self.train()
                train_mean_loss = 0
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
                        train_mean_loss += train_loss.item()
                        for m in metrics.values():
                            m.update(targ, output)
                train_mean_loss /= len(train_loader)
                history['train_loss'].append(train_mean_loss)
                for k, m in metrics.items():
                    history[f'train_{k}'].append(m.compute())
                
                if valid_loader:
                    perform = self.evaluate(loader=valid_loader,
                                            metrics=metrics,
                                            criterion=criterion)
                    for k, v in perform.items():
                        history[k].append(v)
    
    @torch.no_grad()
    def evaluate(self,
                 loader: DataLoader,
                 metrics: Dict,
                 criterion: nn.Module = None,
                 ) -> Dict:
        perform = {}
        
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        self.eval()
        for m in metrics.values():
            m.reset()
        
        mean_loss = 0
        for inp, targ in tqdm(loader, desc=f'eval'):
            inp = inp.to(self.device)
            targ = targ.to(self.device)
            output = self.forward(inp)
            loss = criterion(output, targ)
            output = output.argmax(dim=-1)
            mean_loss += loss.item()
            for m in metrics.values():
                m.update(targ, output)
            
        mean_loss /= len(loader)
        perform['valid_loss'] = mean_loss
        for k, m in metrics.items():
            perform[f'valid_{k}'] = m.compute()
        
        return perform
    
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
    
    @torch.no_grad()
    def predict_torch(self,
                      x: torch.Tensor,
                      ) -> torch.Tensor:
        return self.forward(x)
    
    @property
    def device(self):
        raise NotImplementedError