from Decoder import Decoder
from Encoder import Encoder

import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.Encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.Decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterization(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return (mean + eps * std)

    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, log_var)

        y = self.Decoder(z)
        return (y, mean, log_var)