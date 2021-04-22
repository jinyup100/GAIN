import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path

from data.mnist_data import MaskedMNIST, BlockMaskedMNIST
from data.dataset_loader import CustomDataset
from models.model import GAIN


def main():
    # 1. Load Training Data
    train_data = BlockMaskedMNIST(block_len=7)

    image_data = []
    mask_data = []

    for i in range(len(train_data)):
        image_data.append(train_data[i][0].squeeze())
        mask_data.append(train_data[i][1])

    for i in range(len(image_data)):
        image_data[i][mask_data[i]] = 0

    for i in range(len(mask_data)):
        mask_data[i] = 1 - mask_data[i]

    MNIST_train_dataset = CustomDataset(image_data, mask_data, 28, batch_size=3, hint_rate=0.9)
    train_dataloader = DataLoader(MNIST_train_dataset, batch_size=3, shuffle=True)

    x, m, h = MNIST_train_dataset[0]
    """
    plt.subplot(1, 3, 1)
    plt.imshow(x.cpu().numpy(), cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(m.cpu().numpy(), cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(h, cmap='gray')
    plt.show()
    """

    # 2. Checkpoint and Logger

    checkpoint_path = Path(f'./output')
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(dirpath=str(checkpoint_path),
                                          save_top_k=1,
                                          verbose=True,
                                          monitor='g_loss',
                                          mode='min')

    checkpoint_callback_d = ModelCheckpoint(dirpath=str(checkpoint_path),
                                            save_top_k=1,
                                            verbose=True,
                                            monitor='d_loss',
                                            mode='min')

    # 3. Load Model
    batch_size = 3
    feature_num = 28
    hidden_feature_num = int(feature_num)
    alpha = 500
    hint_rate = 0.9
    model = GAIN(batch_size, feature_num, hidden_feature_num, alpha, hint_rate)

    # 4. Trainer
    logger_path = Path(f'./tb_logs')
    logger_path.mkdir(parents=True, exist_ok=True)
    logger = TensorBoardLogger(save_dir=logger_path)

    trainer = pl.Trainer(callbacks=[checkpoint_callback],
                         gpus=1,
                         logger=logger,
                         max_epochs=1000,
                         )

    # 5. Start Training
    trainer.fit(model.double(), train_dataloader)


if __name__ == '__main__':
    main()




