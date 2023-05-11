
import torch
from sklearn.metrics import (accuracy_score,
                             recall_score,
                             precision_score,
                             f1_score)


class BaseMetrics():
    def __init__(self):
        self.reset()
    
    def update(self):
        raise NotImplementedError()

    def compute(self):
        raise NotImplementedError()
    
    def reset(self):
        raise NotImplementedError()
    

class Accuracy(BaseMetrics):
    def __init__(self) -> None:
        self.reset()
        
    def update(self, 
               label: torch.Tensor,
               pred: torch.Tensor,
               ) -> None:
        self.labels = (torch.concat([self.labels, label]) 
                       if self.labels is not None else label)
        self.preds = (torch.concat([self.preds, pred])
                      if self.preds is not None else pred)
    
    def compute(self, **kwargs):
        return accuracy_score(self.labels.detach().cpu(),
                              self.preds.detach().cpu(),
                              **kwargs)
    
    def reset(self):
        self.labels = None
        self.preds = None
        
    def __call__(self, 
                 labels: torch.Tensor,
                 preds: torch.Tensor,
                 **kwargs,
                 ) -> float:
        self.labels = labels
        self.preds = preds
        return self.compute(**kwargs)


class Recall(BaseMetrics):
    def __init__(self,
                 average: str = 'macro',
                 zero_division: int = 0) -> None:
        self.reset()
        self.average = average
        self.zero_division = zero_division
        
    def update(self, 
               label: torch.Tensor,
               pred: torch.Tensor,
               ) -> None:
        self.labels = (torch.concat([self.labels, label]) 
                       if self.labels is not None else label)
        self.preds = (torch.concat([self.preds, pred])
                      if self.preds is not None else pred)
    
    def compute(self, **kwargs):
        return recall_score(self.labels.detach().cpu(),
                            self.preds.detach().cpu(),
                            average=self.average,
                            zero_division=self.zero_division,
                            **kwargs)
    
    def reset(self):
        self.labels = None
        self.preds = None
        
    def __call__(self, 
                 labels: torch.Tensor,
                 preds: torch.Tensor,
                 **kwargs
                 ) -> float:
        self.labels = labels
        self.preds = preds
        return self.compute(**kwargs)
    
    
class Precision(BaseMetrics):
    def __init__(self,
                 average: str = 'macro',
                 zero_division: int = 0) -> None:
        self.reset()
        self.average = average
        self.zero_division = zero_division
        
    def update(self, 
               label: torch.Tensor,
               pred: torch.Tensor,
               ) -> None:
        self.labels = (torch.concat([self.labels, label]) 
                       if self.labels is not None else label)
        self.preds = (torch.concat([self.preds, pred])
                      if self.preds is not None else pred)
    
    def compute(self, **kwargs):
        return precision_score(self.labels.detach().cpu(),
                               self.preds.detach().cpu(),
                               average=self.average,
                               zero_division=self.zero_division,
                               **kwargs)
    
    def reset(self):
        self.labels = None
        self.preds = None
        
    def __call__(self, 
                 labels: torch.Tensor,
                 preds: torch.Tensor,
                 **kwargs
                 ) -> float:
        self.labels = labels
        self.preds = preds
        return self.compute(**kwargs)


class F1Score(BaseMetrics):
    def __init__(self,
                 average: str = 'macro',
                 zero_division: int = 0) -> None:
        self.reset()
        self.average = average
        self.zero_division = zero_division
        
    def update(self, 
               label: torch.Tensor,
               pred: torch.Tensor,
               ) -> None:
        self.labels = (torch.concat([self.labels, label]) 
                       if self.labels is not None else label)
        self.preds = (torch.concat([self.preds, pred])
                      if self.preds is not None else pred)
    
    def compute(self, **kwargs):
        return f1_score(self.labels.detach().cpu(),
                        self.preds.detach().cpu(),
                        average=self.average,
                        zero_division=self.zero_division,
                        **kwargs)
    
    def reset(self):
        self.labels = None
        self.preds = None
        
    def __call__(self, 
                 labels: torch.Tensor,
                 preds: torch.Tensor,
                 **kwargs
                 ) -> float:
        self.labels = labels
        self.preds = preds
        return self.compute(**kwargs)
    