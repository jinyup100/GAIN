import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
from utils.custom_callback import TrainingCallback
from data.mnist_data import TrainMaskedMNIST, TrainBlockMaskedMNIST, TestMaskedMNIST, TestBlockMaskedMNIST
from data.dataset_loader import CustomDataset
from models.model import GAIN


def main():
    # 1. Load Training Data
    train_data = TrainBlockMaskedMNIST(block_len=7)
    #test_data = TestBlockMaskedMNIST(block_len=7)

    train_image_data = []
    train_mask_data = []
    test_image_data = []
    test_mask_data = []

    for i in range(len(train_data)):
        train_image_data.append(train_data[i][0].squeeze().detach().numpy())
        single_train_mask_data = 1 - train_data[i][1].detach().numpy()
        train_mask_data.append(single_train_mask_data)

    batch_size = 120
    feature_num = 28
    hidden_feature_num = int(feature_num)
    alpha = 1000
    hint_rate = 0.9
    miss_rate = 0.2

    MNIST_train_dataset = CustomDataset(train_image_data, train_mask_data, feature_num, batch_size, hint_rate, miss_rate)
    train_dataloader = DataLoader(MNIST_train_dataset, batch_size, shuffle=True)

    x, m, h = MNIST_train_dataset[0]
    plt.subplot(1, 3, 1)
    plt.imshow(x.flatten().reshape(feature_num, feature_num).cpu().numpy(), cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(m.flatten().reshape(feature_num, feature_num).cpu().numpy(), cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(h.flatten().reshape(feature_num, feature_num).cpu().numpy(), cmap='gray')
    plt.show()

    # 2. Checkpoint and Logger
    checkpoint_path = Path(f'./output')
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(dirpath=str(checkpoint_path),
                                          save_top_k=4,
                                          verbose=True,
                                          monitor='g_loss',
                                          mode='min')

    checkpoint_callback_d = ModelCheckpoint(dirpath=str(checkpoint_path),
                                            save_top_k=3,
                                            verbose=True,
                                            monitor='d_loss',
                                            mode='min')

    train_callback = TrainingCallback(train_dataloader, 3)

    # 3. Load Model
    model = GAIN(batch_size, feature_num, hidden_feature_num, alpha, hint_rate)

    # 4. Trainer
    logger_path = Path(f'./tb_logs')
    logger_path.mkdir(parents=True, exist_ok=True)
    logger = TensorBoardLogger(save_dir=logger_path)

    trainer = pl.Trainer(callbacks=[checkpoint_callback, train_callback],
                         gpus=1,
                         logger=logger,
                         max_epochs=1000,
                         progress_bar_refresh_rate=1
                         )

    # 5. Start Training
    trainer.fit(model.double(), train_dataloader)


if __name__ == '__main__':
    main()




