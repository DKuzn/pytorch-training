# pytorch_training

This repository contains function to training PyTorch models.

## How to use
Clone this repository to your project.

Example use case:

```python
from pytorch_training import training
from torchvision.models import resnet18
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch

if __name__ == '__main__':
    ds_train = MNIST(root='./', train=True)
    ds_test = MNIST(root='./', train=False)
    dl_train = DataLoader(ds_train, batch_size=32, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=32)
    model = resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, 10)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    accuracy = lambda preds, y_batch: (preds.argmax(dim=1) == y_batch).float().mean().data.cpu()

    training(dl_train, dl_test, model, optimizer, loss_function, accuracy, epochs=100)
```