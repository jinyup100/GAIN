import numpy as np

from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, normalized_data_x, data_m, number_of_features, batch_size, hint_rate):
        self.normalized_x_data = normalized_data_x.copy()
        self.m_data = data_m.copy()
        self.batch_size = batch_size
        self.hint_rate = hint_rate
        self.number_of_features = number_of_features

    def __binary_sampler(self, p, rows, cols):
        unif_random_matrix = np.random.uniform(0., 1., size=[rows, cols])
        binary_random_matrix = 1 * (unif_random_matrix < p)
        return binary_random_matrix

    def __len__(self):
        return len(self.normalized_x_data)

    def __getitem__(self, index):
        x = self.normalized_x_data[index]
        m = self.m_data[index]
        z = np.random.uniform(0, 0.01, size=[x.shape[0], x.shape[1]])

        h_sample = self.__binary_sampler(self.hint_rate, x.shape[0], x.shape[1])
        h = (1 - m) * h_sample
        h = 1 - h
        x = x + (1 - m) * z

        x = x.double()
        m = m.double()
        h = h.double()

        return x, m, h