# pytorch_training/training.py
#
# Copyright (C) 2021 Дмитрий Кузнецов
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Module training.

This module contains training function.
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Module
from torch.optim import Optimizer
from torch import Tensor
from typing import Dict, Union, List, Tuple, Callable
from datetime import datetime
from tqdm import tqdm
import os


def training(dl_train: DataLoader,
             dl_test: DataLoader,
             model: Module,
             optimizer: Optimizer,
             loss_function: Callable[[Tensor, Tensor], Tensor],
             accuracy: Callable[[Tensor, Tensor], Union[Tensor, List[Tensor], Tuple[Tensor]]],
             epochs: int = 100,
             checkpoint_best: str = f'weights{os.sep}best_weights.pt',
             checkpoint_last: str = f'weights{os.sep}last_weights.pt',
             log_path: str = 'logs',
             checkpoint: Dict[str, Dict[str, Union[Tensor, List, str]]] = None) -> None:
    """The function to training PyTorch models with TensorBoard logging.

    Args:
        dl_train: PyTorch DataLoader.
        dl_test: PyTorch DataLoader.
        model: PyTorch model.
        optimizer: PyTorch optimizer.
        loss_function: PyTorch built-in or custom loss function.
        accuracy: Custom function to accuracy calculation.
        epochs: Count of epochs to training.
        checkpoint_best: Path to save best model weights.
        checkpoint_last: Path to save last model weights.
        log_path: Path to TensorBoard logging.
        checkpoint: Last weights loaded with torch.load().

    Return:
        None
    """

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
        log_path = os.path.join(log_path, f'{datetime.now().strftime("%Y%m%d-%H%M%S")}')
        writer = SummaryWriter(log_dir=log_path)

        init_epoch = 0
        less_loss = 1e+10

    checkpoint_best_dir = os.path.dirname(checkpoint_best)
    checkpoint_last_dir = os.path.dirname(checkpoint_last)

    if not os.path.exists(checkpoint_best_dir) and checkpoint_best_dir != '':
        os.mkdir(checkpoint_best_dir)

    if not os.path.exists(checkpoint_last_dir) and checkpoint_last_dir != '':
        os.mkdir(checkpoint_last_dir)

    for epoch in range(init_epoch, epochs, 1):
        print(f'\rEpoch {epoch + 1}/{epochs}')
        train_loss = 0.0
        train_accuracy = 0.0

        train_bar = tqdm(dl_train, desc='Training', total=len(dl_train), unit='batch', ncols=80)

        for x_batch, y_batch in train_bar:
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

        print(f'\rtrain_loss: {round(float(train_loss), 4)} - train_accuracy: {round(float(train_accuracy), 4)}')

        test_loss = 0.0
        test_accuracy = 0.0

        test_bar = tqdm(dl_test, desc='Testing', total=len(dl_test), unit='batch', ncols=80)

        for x_test_batch, y_test_batch in test_bar:
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

        print(f'\rtest_loss: {round(float(test_loss), 4)} - test_accuracy: {round(float(test_accuracy), 4)}')

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
