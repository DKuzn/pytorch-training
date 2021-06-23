import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Module
from torch.optim import Optimizer
from torch import Tensor
from typing import Dict, Union, List
from datetime import datetime
from tqdm import tqdm


def training(dl_train: DataLoader,
             dl_test: DataLoader,
             model: Module,
             optimizer: Optimizer,
             loss_function,
             accuracy,
             epochs: int = 100,
             checkpoint_best: str = 'weights/best_weights.pt',
             checkpoint_last: str = 'weights/last_weights.pt',
             log_path: str = 'logs',
             checkpoint: Dict[str, Dict[str, Union[Tensor, List, str]]] = None):

    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    loss_function = loss_function
    optimizer = optimizer

    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        model.train()
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        init_epoch = checkpoint['epoch']
        less_loss = checkpoint['less_loss']
        log_path = str(checkpoint['log_path'])
        writer = SummaryWriter(log_path)

    else:
        log_path += f'/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        writer = SummaryWriter(log_dir=log_path)

        init_epoch = 0
        less_loss = 1e+10

    for epoch in range(init_epoch, epochs, 1):
        print(f'Epoch {epoch + 1}/{epochs}')
        train_loss = 0.0
        train_accuracy = 0.0

        for x_batch, y_batch in tqdm(dl_train, total=len(dl_train)):
            optimizer.zero_grad()

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model.forward(x_batch)
            train_accuracy += accuracy(preds, y_batch)

            loss_value = loss_function(preds, y_batch)
            train_loss += loss_value

            loss_value.backward()

            optimizer.step()

        train_loss /= len(dl_train)
        train_accuracy /= len(dl_train)

        print(f'\r - train_loss: {round(float(train_loss), 4)} - train_accuracy: {round(float(train_accuracy), 4)}')

        test_loss = 0.0
        test_accuracy = 0.0

        for x_test_batch, y_test_batch in tqdm(dl_test, total=len(dl_test)):
            x_test_batch = x_test_batch.to(device)
            y_test_batch = y_test_batch.to(device)

            test_preds = model.forward(x_test_batch)
            test_accuracy += accuracy(test_preds, y_test_batch)

            test_loss += loss_function(test_preds, y_test_batch).data.cpu()

        test_loss /= len(dl_test)
        test_accuracy /= len(dl_test)

        writer.add_scalars('Loss', {'Train': train_loss,
                                    'Test': test_loss}, epoch)

        writer.add_scalars('Accuracy', {'Train': train_accuracy,
                                        'Test': test_accuracy}, epoch)

        writer.flush()

        print(f'\r - test_loss: {round(float(test_loss), 4)} - test_accuracy: {round(float(test_accuracy), 4)}')

        if test_loss <= less_loss:
            torch.save(model.state_dict(), checkpoint_best)
            print(f'Test loss improve from {less_loss} to {test_loss}. Saving to {checkpoint_best}')
            less_loss = test_loss
        else:
            print(f'Test loss did not improve from {less_loss}.')

        print(f'Epoch weights saved to {checkpoint_last}')

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'less_loss': less_loss,
            'log_path': log_path
        }, checkpoint_last)
