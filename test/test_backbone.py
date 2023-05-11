
from pathlib import Path

import cv2
import torchvision

from vismo.backbone import *
from vismo.metrics import (Accuracy,
                           Recall, 
                           Precision,
                           F1Score)


if __name__ == '__main__':
    model = LeNet5().to('cuda:0')
    trs = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((32, 32),
                                      antialias=True),
    ])
    train_data = torchvision.datasets.MNIST(root='./data',
                                            train=True,
                                            download=True,
                                            transform=trs)
    test_data = torchvision.datasets.MNIST(root='./data',
                                           train=False,
                                           download=True,
                                           transform=trs)
    train_loader = DataLoader(train_data,
                              batch_size=128,
                              shuffle=True,
                              num_workers=4)
    test_loader = DataLoader(test_data,
                             batch_size=128,
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
              valid_loader=test_loader,
              fp16=True,
              on_after_epoch=callback)
    
    paths = Path('data/MNIST/samples').glob('*.png')
    imgs = [cv2.imread(str(p), 0)[:, :, None] for p in paths]
    imgs = np.stack(imgs)
    outputs = model.predict(imgs)
    print(outputs)
    