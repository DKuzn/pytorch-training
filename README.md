# pytorch_training

This repository contains the function to training PyTorch models.

## How to use
Clone this repository to your project.

Example use case:

```python
from pytorch_training import training
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch

if __name__ == '__main__':
    ds_train = CIFAR10(root='./', train=True, transform=ToTensor())
    ds_test = CIFAR10(root='./', train=False, transform=ToTensor())
    dl_train = DataLoader(ds_train, batch_size=32, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=32)
    model = resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, 10)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    accuracy = lambda preds, y_batch: (preds.argmax(dim=1) == y_batch).float().mean().data.cpu()

    training(dl_train, dl_test, model, optimizer, loss_function, accuracy, epochs=100)
```