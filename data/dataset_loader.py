import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, data_x, data_m, feature_num, batch_size, hint_rate, miss_rate):
        self.normalized_x_data = data_x.copy()
        self.m_data = data_m.copy()
        self.batch_size = batch_size
        self.hint_rate = hint_rate
        self.miss_rate = miss_rate
        self.feature_num = feature_num
        self.number_of_features = feature_num**2

    def __binary_sampler(self, p, rows, cols):
        unif_random_matrix = np.random.uniform(0., 1., size=[rows, cols])
        binary_random_matrix = 1 * (unif_random_matrix < p)
        return binary_random_matrix

    def __sample_M(self, m, n, p):
        A = np.random.uniform(0., 1., size=[m, n])
        B = A > p
        C = 1. * B
        return C

    def __sample_Z(self, m, n):
        return np.random.uniform(0., 1., size=[m, n])

    def __len__(self):
        return len(self.normalized_x_data)

    def __getitem__(self, index):
        x = self.normalized_x_data[index]
        x = x.reshape(1, self.number_of_features)

        z = self.__sample_Z(1, self.number_of_features)

        m = self.m_data[index]
        m = m.reshape(1, self.number_of_features)

        h_sample = self.__sample_M(1, self.number_of_features, 1 - self.hint_rate)
        h = m * h_sample + 0.5*(1-h_sample)

        new_x = m*x + (1-m)*z

        new_x = new_x.reshape(1, self.feature_num, self.feature_num)
        m = m.reshape(1, self.feature_num, self.feature_num)
        h = h.reshape(1, self.feature_num, self.feature_num)

        new_x = torch.DoubleTensor(new_x).flatten(start_dim=0, end_dim=1)
        m = torch.DoubleTensor(m).flatten(start_dim=0, end_dim=1)
        h = torch.DoubleTensor(h).flatten(start_dim=0, end_dim=1)

        return new_x, m, h