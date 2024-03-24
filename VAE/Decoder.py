import torch.nn as nn
import torch

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.layer1 = nn.Linear(latent_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.generator = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        latent = self.LeakyReLU(self.layer1(x))
        latent = self.LeakyReLU(self.layer2(latent))

        result = self.generator(latent)
        #for generator img using signmoid
        return (torch.sigmoid(result))
