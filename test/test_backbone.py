
from pathlib import Path
from PIL import Image

import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torch.utils.data import DataLoader, Dataset

from vismo.backbone import *
from vismo.metrics import (Accuracy,
                           Recall, 
                           Precision,
                           F1Score)


class ImageNet(Dataset):
    def __init__(self, 
                 paths,
                 targs,
                 transform=None):
        self.paths = paths
        self.targs = targs
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        p = self.paths[index]
        img = np.array(Image.open(p).convert('RGB'), dtype=np.uint8)
        targ = self.targs[index]
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, targ
        

if __name__ == '__main__':
    model = VGG16_BN(num_classes=10).to('cuda:0')
    print(model.device)
    trs = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((32, 32),
                                      antialias=True),
    ])
    train_data = torchvision.datasets.CIFAR10(root='./data',
                                              train=True,
                                              download=True,
                                              transform=trs)
    valid_data = torchvision.datasets.CIFAR10(root='./data',
                                             train=False,
                                             download=True,
                                             transform=trs)
    
    # CLASS_NAMES = [d.name for d in Path('/data/imagenet-mini/train').glob('*')]
    # train_paths = list(Path('/data/imagenet-mini/train').glob('*/*.JPEG'))
    # train_labels = [CLASS_NAMES.index(p.parent.name) for p in train_paths]
    # valid_paths = list(Path('/data/imagenet-mini/val').glob('*/*.JPEG'))
    # valid_labels = [CLASS_NAMES.index(p.parent.name) for p in valid_paths]
    
    # train_data = ImageNet(paths=train_paths,
    #                       targs=train_labels,
    #                       transform=trs)
    # valid_data = ImageNet(paths=valid_paths,
    #                       targs=valid_labels,
    #                       transform=trs)
    
    train_loader = DataLoader(train_data,
                              batch_size=256,
                              shuffle=True,
                              num_workers=4)
    valid_loader = DataLoader(valid_data,
                             batch_size=256,
                             num_workers=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=1e-2)
    def callback(t, history):
        log = f'{t:.2f}s, '
        log += f"train loss: {history['train_loss'][-1]:.4f}, "
        log += f"train f1: {history['train_f1'][-1]:.4f}, "
        log += f"valid loss: {history['valid_loss'][-1]:.4f}, "
        log += f"valid f1: {history['valid_f1'][-1]:.4f}, "
        print(log)
        
    model.fit(train_loader=train_loader,
              criterion=criterion,
              optimizer=optimizer,
              epochs=15,
              metrics={
                  'acc': Accuracy(),
                  'prec': Precision(),
                  'rec': Recall(),
                  'f1': F1Score(),
              },
              valid_loader=valid_loader,
              fp16=True,
              on_after_epoch=callback,
              multi_gpus=[0, 1, 2])
    
    paths = list(Path('data/cifar-10-batches-py/samples').glob('*.png'))
    imgs = [np.array(Image.open(p), dtype=np.uint8)
            for p in paths]
    labels = [int(p.stem.split('_')[-1])
              for p in paths]
    imgs = np.stack(imgs)
    outputs = model.predict(imgs)
    print(outputs.argmax(dim=-1), labels)
    