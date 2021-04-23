import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from pytorch_lightning.callbacks import Callback, EarlyStopping
from pathlib import Path
from tqdm import tqdm


class TrainingCallback(Callback):
    def __init__(self, MNIST_test_dataloader, number_of_displays):
        super(TrainingCallback, self).__init__()
        self.MNIST_test_dataloader = MNIST_test_dataloader
        self.number_of_displays = number_of_displays

    def on_epoch_end(self, trainer, pl_module) -> None:
        print("----------Training Callback------------")
        x, m, h = next(iter(self.MNIST_test_dataloader))
        x = x.cuda()
        m = m.cuda()
        generator_sample = trainer.model.forward(x, m)

        n_rows = self.number_of_displays
        n_columns = 3
        n_place = 1

        for i in tqdm(range(1, self.number_of_displays+1)):
            imputed_data = m[i]*x[i] + (1-m[i])*generator_sample[i]

            plt.subplot(n_rows, n_columns, n_place)
            plt.imshow(x[i].cpu().detach().numpy(), cmap='gray')
            n_place += 1

            plt.subplot(n_rows, n_columns, n_place)
            plt.imshow(m[i].cpu().detach().numpy(), cmap='gray')
            n_place += 1

            plt.subplot(n_rows, n_columns, n_place)
            plt.imshow(imputed_data.cpu().detach().numpy(), cmap='gray')
            n_place += 1

        plt.show()
