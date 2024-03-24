
import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torch.optim import Adam

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from VAE import VAE

if __name__ == '__main__':
    #Model Hyperparameters
    dataset_path = '../datasets'

    cuda = True
    DEVICE = torch.device("cuda" if cuda else "cpu")


    batch_size = 100
    x_dim  = 784
    hidden_dim = 400
    latent_dim = 200

    lr = 1e-3

    epochs = 30


    mnist_transform = transforms.Compose([
            transforms.ToTensor(),
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True} 

    train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
    test_dataset  = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False, **kwargs)


    model = VAE(x_dim, hidden_dim, latent_dim).to(DEVICE)



    BCE_loss = nn.BCELoss()

    def loss_function(x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

        return reproduction_loss + KLD


    optimizer = Adam(model.parameters(), lr=lr)
    print("Start training VAE...")
    model.train()

    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
        
    print("Finish!!")


    #inference

    import matplotlib.pyplot as plt

    model.eval()

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(tqdm(test_loader)):
            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)
            
            x_hat, _, _ = model(x)


            break

    def show_image(x, idx):
        x = x.view(batch_size, 28, 28)

        fig = plt.figure()
        plt.imshow(x[idx].cpu().numpy())

    show_image(x, idx=0)
    show_image(x_hat, idx=0)