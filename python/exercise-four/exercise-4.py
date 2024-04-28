import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the Generator network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Define the Discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def train_gan(batch_size=32, num_epochs=100, device='cpu'):
    # Set up device
    device = torch.device(device)

    # Load MNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Initialize networks
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Define loss function and optimizers
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)

    # Training loop
    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(train_loader):
            real_images = real_images.view(-1, 784).to(device)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train Discriminator
            optimizer_d.zero_grad()
            # Real images
            output = discriminator(real_images)
            loss_d_real = criterion(output, real_labels)
            loss_d_real.backward()
            # Fake images
            z = torch.randn(batch_size, 100).to(device)
            fake_images = generator(z)
            output = discriminator(fake_images.detach())
            loss_d_fake = criterion(output, fake_labels)
            loss_d_fake.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            z = torch.randn(batch_size, 100).to(device)
            fake_images = generator(z)
            output = discriminator(fake_images)
            loss_g = criterion(output, real_labels)
            loss_g.backward()
            optimizer_g.step()

        # Print progress
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss D Real: {loss_d_real.item()}, Loss D Fake: {loss_d_fake.item()}, Loss G: {loss_g.item()}')

# Example usage
train_gan(batch_size=32, num_epochs=100)


# Bugs:

# 1. Structural Bug: The structural bug arises when the batch size is changed from 32 to 64. This happens because the all_samples and all_samples_labels tensors are concatenated during discriminator training.

#    - When batch_size is 32, the concatenation results in tensors of size (batch_size * 2, 1), which works for both loss_function (BCELoss) and the discriminator output (output_discriminator).
#    - However, when batch_size is changed to 64, the concatenation leads to tensors of size (batch_size * 2, 1), which becomes (128, 1). This size mismatch triggers the error "Using a target size (torch.Size([128, 1])) that is different to the input size (torch.Size([96, 1]))".

# 2. Cosmetic Bug: This bug is less critical but affects the output display. The code shows generated images after every epoch, but it doesn't clear the previous display. This results in an accumulation of figures on the screen.

# Fixes:

# 1. Structural Bug Fix: To ensure the concatenated tensors have compatible sizes for the loss function, we can reshape the real_samples_labels and generated_samples_labels tensors before concatenation:


