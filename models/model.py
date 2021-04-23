import torch
import torch.nn as nn
import pytorch_lightning as pl

from collections import OrderedDict


class Generator(pl.LightningModule):
    def __init__(self, number_of_features, hidden_state_features):
        super(Generator, self).__init__()

        self.g_layer_1 = nn.Linear(number_of_features, hidden_state_features)
        torch.nn.init.xavier_uniform_(self.g_layer_1.weight)
        self.g_layer_1_relu = nn.ReLU()

        self.g_layer_2 = nn.Linear(hidden_state_features, hidden_state_features)
        torch.nn.init.xavier_uniform_(self.g_layer_1.weight)
        self.g_layer_2_relu = nn.ReLU()

        self.g_layer_3 = nn.Linear(hidden_state_features, hidden_state_features)
        torch.nn.init.xavier_uniform_(self.g_layer_3.weight)
        self.g_layer_3_sigmoid = nn.Sigmoid()

    def forward(self, x, mask):
        inputs = torch.cat([x, mask], axis=2)
        g_h_1_output = self.g_layer_1_relu(self.g_layer_1(inputs))
        g_h_2_output = self.g_layer_2_relu(self.g_layer_2(g_h_1_output))
        g_prob = self.g_layer_3_sigmoid(self.g_layer_3(g_h_2_output))
        return g_prob


class Discriminator(pl.LightningModule):
    def __init__(self, number_of_features, hidden_state_features):
        super(Discriminator, self).__init__()
        self.d_layer_1 = nn.Linear(number_of_features, hidden_state_features)
        torch.nn.init.xavier_uniform_(self.d_layer_1.weight)
        self.d_layer_1_relu = nn.ReLU()

        self.d_layer_2 = nn.Linear(hidden_state_features, hidden_state_features)
        torch.nn.init.xavier_uniform_(self.d_layer_1.weight)
        self.d_layer_2_relu = nn.ReLU()

        self.d_layer_3 = nn.Linear(hidden_state_features, hidden_state_features)
        torch.nn.init.xavier_uniform_(self.d_layer_3.weight)
        self.d_layer_3_sigmoid = nn.Sigmoid()

    def forward(self, x, hint):
        inputs = torch.cat([x, hint], axis=2)
        d_h_1_output = self.d_layer_1_relu(self.d_layer_1(inputs))
        d_h_2_output = self.d_layer_2_relu(self.d_layer_2(d_h_1_output))
        d_prob = self.d_layer_3_sigmoid(self.d_layer_3(d_h_2_output))
        return d_prob


class GAIN(pl.LightningModule):
    def __init__(self, batch_size, number_of_features, hidden_state_features, alpha, hint_rate):
        super(GAIN, self).__init__()
        self.automatic_optimization = False
        self.alpha = alpha
        self.batch_size = batch_size
        self.hint_rate = hint_rate
        self.number_of_features = number_of_features
        self.hidden_state_features = hidden_state_features
        self.generator = Generator(2 * number_of_features, hidden_state_features)
        self.discriminator = Discriminator(2 * number_of_features, hidden_state_features)
        self.lr = 0.0001

    def forward(self, x, m):
        generator_sample = self.generator(x, m)
        return generator_sample

    def _binary_sampler(self, p, rows, cols):
        uniform_random_matrix = torch.rand([rows, cols])
        binary_random_matrix = 1 * (uniform_random_matrix < p)
        return binary_random_matrix

    def _generator_step(self, x, mask, hint):
        mse_loss = nn.MSELoss(reduction="mean")

        generator_sample = self.generator(x, mask)
        x_hat = x*mask + generator_sample * (1 - mask)
        discriminator_prob = self.discriminator(x_hat, hint)

        # G_CE_loss = -torch.mean((1-mask) * torch.log(discriminator_prob + 1e-8))
        G_CE_loss = ((1 - mask) * (discriminator_prob + 1e-8).log()).mean() / (1 - mask).sum()
        G_MSE_loss = mse_loss(mask * x, mask * generator_sample) / mask.sum()
        # G_MSE_loss = torch.mean((mask*x - mask*generator_sample)**2) / torch.mean(mask)

        G_total_loss = G_CE_loss + G_MSE_loss * self.alpha
        return G_total_loss

    def _discriminator_step(self, x, mask, hint):
        bce_loss = nn.BCELoss(reduction="mean")

        generator_sample = self.generator(x, mask)
        x_hat = x*mask + generator_sample * (1 - mask)
        discriminator_prob = self.discriminator(x_hat.detach(), hint)
        # D_total_loss = -torch.mean(mask * torch.log(discriminator_prob + 1e-8) + (1-mask) * torch.log(1. - discriminator_prob + 1e-8))
        D_total_loss = bce_loss(discriminator_prob, mask)
        return D_total_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        g_opt, d_opt = self.optimizers()
        # Batch x and mask
        x, mask, hint = batch

        # Train Generator
        if optimizer_idx == 0:
            g_loss = self._generator_step(x, mask, hint)
            self.log('g_loss', g_loss, prog_bar=True, logger=True, on_step=True)
            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()
            loss = g_loss
            return loss

        if optimizer_idx == 1:
            d_loss = self._discriminator_step(x, mask, hint)
            self.log('d_loss', d_loss, prog_bar=True, logger=True, on_step=True)
            d_loss.backward()
            d_opt.step()
            d_opt.zero_grad()
            loss = d_loss
            return loss

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        return [g_opt, d_opt], []


if __name__ == '__main__':
    batch_size = 12
    number_of_features = 512
    hidden_state_features= 512
    alpha = 10
    hint_rate = 0.8

    model = GAIN(batch_size, number_of_features, hidden_state_features, alpha, hint_rate)

