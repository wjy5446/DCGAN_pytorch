import torch
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from model import Generator, Discriminator
from loss import GANLoss
from utils.misc import load_model, save_model

def run():
    # Dataset
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST('.', transform=transform, download=True)
    dataloader = data.DataLoader(dataset, batch_size=4)
    print("[INFO] Define DataLoader")

    # Define Model
    g = Generator()
    d = Discriminator()
    print("[INFO] Define Model")

    # optimizer, loss
    gan_loss = GANLoss()

    optim_G = optim.Adam(g.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optim_D = optim.Adam(d.parameters(), lr=0.0002, betas=(0.5, 0.999))
    print('[INFO] Define optimizer and loss')

    # train
    num_epoch = 2

    print('[INFO] Start Training!!')
    for epoch in range(num_epoch):
        total_batch = len(dataloader)

        for idx, (image, _) in enumerate(dataloader):
            d.train()
            g.train()

            # fake image 생성
            noise = torch.randn(4, 100, 1, 1)
            output_fake = g(noise)

            # Loss

            d_loss_fake = gan_loss(d(output_fake.detach()), False)
            d_loss_real = gan_loss(d(image), True)
            d_loss = (d_loss_fake + d_loss_real) / 2

            g_loss = gan_loss(d(output_fake), True)

            # update
            optim_G.zero_grad()
            g_loss.backward()
            optim_G.step()

            optim_D.zero_grad()
            d_loss.backward()
            optim_D.step()

            if ((epoch * total_batch) + idx) % 1000 == 0:
                print('Epoch [%d/%d], Iter [%d/%d], D_loss: %.4f, G_loss: %.4f'
                      % (epoch, num_epoch, idx + 1, total_batch, d_loss.item(), g_loss.item()))

                save_model('model', 'GAN', g, {'loss': g_loss.item()})

if __name__ == '__main__':
    run()