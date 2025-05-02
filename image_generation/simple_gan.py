"""
A minimal PyTorch implementation of a Generative Adversarial Network (GAN) for educational purposes.
This script trains a GAN on the MNIST dataset to generate handwritten digit images.

1. **Generator**: Takes a random noise vector (latent space) and generates a fake image.
2. **Discriminator**: Takes an image (real or fake) and outputs a probability that the image is real.
3. **Adversarial training**:
   - Train Discriminator to distinguish real images from Generator's fake images.
   - Train Generator to fool the Discriminator into classifying fake images as real.

Outputs:
- A folder `images/` containing generated image grids after each epoch.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Hyperparameters 
batch_size = 64
latent_dim = 100
learning_rate = 0.0002
num_epochs = 20
img_size = 28
img_channels = 1
sample_interval = 5  # Save images every N epochs

# Create output directory
os.makedirs('images', exist_ok=True)

# Data Loader 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Scale images to [-1, 1]
])
dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model Definitions 
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Input: latent_dim -> Hidden 1
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            # Hidden 1 -> Hidden 2
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # Hidden 2 -> Hidden 3
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # Hidden 3 -> Output image size
            nn.Linear(512, img_channels * img_size * img_size),
            nn.Tanh()  # Output values in [-1, 1]
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(z.size(0), img_channels, img_size, img_size)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_channels * img_size * img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # Output probability [0,1]
        )

    def forward(self, img):
        flat = img.view(img.size(0), -1)
        validity = self.model(flat)
        return validity

# Initialize models and optimizers
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = Generator().to(device)
discriminator = Discriminator().to(device)

adversarial_loss = nn.BCELoss()

g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Training Loop 
for epoch in range(1, num_epochs + 1):
    for real_imgs, _ in dataloader:
        batch_size_curr = real_imgs.size(0)
        real_imgs = real_imgs.to(device)

        # Create labels
        real_labels = torch.ones(batch_size_curr, 1, device=device)
        fake_labels = torch.zeros(batch_size_curr, 1, device=device)

        # Train Discriminator 
        # Zero gradients
        d_optimizer.zero_grad()

        # Loss on real images
        real_loss = adversarial_loss(discriminator(real_imgs), real_labels)

        # Generate fake images
        z = torch.randn(batch_size_curr, latent_dim, device=device)
        fake_imgs = generator(z)

        # Loss on fake images
        fake_loss = adversarial_loss(discriminator(fake_imgs.detach()), fake_labels)

        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        d_optimizer.step()

        #  Train Generator 
        g_optimizer.zero_grad()

        # Try to fool the discriminator: want discriminator(fake_imgs) to be real
        g_loss = adversarial_loss(discriminator(fake_imgs), real_labels)
        g_loss.backward()
        g_optimizer.step()

    # Print progress
    print(f"[Epoch {epoch}/{num_epochs}]  D_loss: {d_loss.item():.4f}  G_loss: {g_loss.item():.4f}")

    # Save sample generated images
    if epoch % sample_interval == 0 or epoch == num_epochs:
        with torch.no_grad():
            test_z = torch.randn(64, latent_dim, device=device)
            gen_imgs = generator(test_z)
            # Rescale images from [-1,1] to [0,1]
            gen_imgs = (gen_imgs + 1) / 2
            save_image(gen_imgs, f"images/epoch_{epoch:03d}.png", nrow=8)

print("Training finished. Check the 'images' folder for generated samples.")
