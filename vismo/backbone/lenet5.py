
import torch
import torch.nn as nn


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
    
    @property
    def device(self):
        return self.c1.weight.device
    
    
if __name__ == '__main__':
    model = LeNet5()
    x = torch.rand(size=(4, 1, 32, 32))
    output = model(x)
    print(output)
    