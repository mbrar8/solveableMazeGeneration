import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from maze_dataset import MazeDataset
import torchvision
# from torchvision.transforms import ToTensor

# Define the Variational Autoencoder (VAE) architecture
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate the size of the flattened output from the encoder
        self.encoder_output_size = self._get_encoder_output_size(input_dim)

        self.fc_mu = nn.Linear(self.encoder_output_size, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_output_size, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, self.encoder_output_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def _get_encoder_output_size(self, input_dim):
        # Forward pass to determine the output size of the encoder
        with torch.no_grad():
            fake_input = torch.zeros((1, 3, input_dim, input_dim))
            encoder_output = self.encoder(fake_input)
            return encoder_output.size(1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        # Encode
        x = self.encoder(x)

        # Compute latent space parameters
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        x = self.decoder_input(z)
        x = x.view(-1, 256, 2, 2)  # Reshape to match expected input of decoder
        x = self.decoder(x)
        return x, mu, logvar

# Define training function
def train_vae(vae, dataloader, epochs=10, lr=1e-3):
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss for reconstruction

    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            inputs = batch
            breakpoint()
            outputs, mu, log_var = vae(inputs)

            # Flatten inputs for BCELoss
            inputs = inputs.view(-1, 3, 365, 365)

            # Reconstruction loss
            reconstruction_loss = criterion(outputs, inputs)

            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            # Total loss
            loss = reconstruction_loss + kl_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

# Example usage
if __name__ == "__main__":
    
    input_dim = 365  # Assuming square images
    latent_dim = 64  # Dimension of the latent space

    # Initialize VAE
    vae = VAE(input_dim, latent_dim)

    # Train VAE
    dataset = MazeDataset(transform=torchvision.transforms.ToTensor())
    train_set, val_set = random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=32, shuffle=True)
    train_vae(vae, train_dataloader)
