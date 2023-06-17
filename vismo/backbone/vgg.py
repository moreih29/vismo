from typing import Dict, List, Union, cast

import torch
import torch.nn as nn

from .base import BaseBackbone


__all__ = ['VGG11', 'VGG11_BN', 'VGG13', 'VGG13_BN', 
           'VGG16', 'VGG16_BN', 'VGG19', 'VGG19_BN']

cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGG(BaseBackbone):
    def __init__(
        self, 
        features: nn.Module, 
        num_classes: int = 1000, 
        init_weights: bool = True, 
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward_feature(self, 
                        x: torch.Tensor,
                        ) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def forward_head(self, 
                     x: torch.Tensor,
                     ) -> torch.Tensor:
        x = self.classifier(x)
        return x
    
    @property
    def device(self):
        return self.classifier._modules.get('0').weight.device
    

def make_layers(cfg: List[Union[str, int]],
                batch_norm: bool = False,
                ) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG11(VGG):
    def __init__(self, 
                 num_classes: int = 1000,
                 **kwargs) -> None:
        super().__init__(make_layers(cfgs['A'], False),
                         num_classes=num_classes,
                         **kwargs)
        

class VGG11_BN(VGG):
    def __init__(self, 
                 num_classes: int = 1000,
                 **kwargs) -> None:
        super().__init__(make_layers(cfgs['A'], True),
                         num_classes=num_classes,
                         **kwargs)
        

class VGG13(VGG):
    def __init__(self, 
                 num_classes: int = 1000,
                 **kwargs) -> None:
        super().__init__(make_layers(cfgs['B'], False),
                         num_classes=num_classes,
                         **kwargs)
        
        
class VGG13_BN(VGG):
    def __init__(self, 
                 num_classes: int = 1000,
                 **kwargs) -> None:
        super().__init__(make_layers(cfgs['B'], True),
                         num_classes=num_classes,
                         **kwargs)
        

class VGG16(VGG):
    def __init__(self, 
                 num_classes: int = 1000,
                 **kwargs) -> None:
        super().__init__(make_layers(cfgs['D'], False),
                         num_classes=num_classes,
                         **kwargs)
        

class VGG16_BN(VGG):
    def __init__(self, 
                 num_classes: int = 1000,
                 **kwargs) -> None:
        super().__init__(make_layers(cfgs['D'], True),
                         num_classes=num_classes,
                         **kwargs)
        

class VGG19(VGG):
    def __init__(self, 
                 num_classes: int = 1000,
                 **kwargs) -> None:
        super().__init__(make_layers(cfgs['E'], False),
                         num_classes=num_classes,
                         **kwargs)
        
    
class VGG19_BN(VGG):
    def __init__(self, 
                 num_classes: int = 1000,
                 **kwargs) -> None:
        super().__init__(make_layers(cfgs['E'], True),
                         num_classes=num_classes,
                         **kwargs)