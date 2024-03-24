import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        latent = self.LeakyReLU(self.layer1(x))
        latent = self.LeakyReLU(self.layer2(latent))

        mean = self.mean(latent)
        log_var = self.var(latent)

        return (mean, log_var)
    
