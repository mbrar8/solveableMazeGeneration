# imported from https://github.com/AntixK/PyTorch-VAE/tree/master

import torch
#from models import BaseVAE
from torch import nn
from torch.nn import functional as F
#from .types_ import *


class VanillaVAE(nn.Module):


    def __init__(self,
                 in_channels,
                 latent_dim,
                 hidden_dims,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [8, 16]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride=2),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        #self.encoder = nn.Sequential(*modules)
        self.encode1 = nn.Conv2d(3, 8, kernel_size=3, stride=2)
        self.encode2 = nn.Conv2d(8, 16, kernel_size=3, stride=2)
        self.encode3 = nn.Conv2d(16, 32, kernel_size=3, stride=2)

        self.fc_mu = nn.Linear(3872, latent_dim)
        self.fc_var = nn.Linear(3872, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, 3872)
        self.decode1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, output_padding=1)
        self.decode2 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, output_padding=1)
        self.decode3 = nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, output_padding=1)
        self.decode4 = nn.Conv2d(8, 3, kernel_size=3)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2),
                                       #padding=1,
                                       #output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )


      
        #self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2),
                                               #padding=1,
                                               #output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3),
                            nn.Tanh())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        # print("ENCODING")
        result = self.encode1(input)
        # print(result.shape)
        result = nn.BatchNorm2d(8)(result)
        result = nn.LeakyReLU()(result)
        result = self.encode2(result)
        # print(result.shape)
        result = nn.BatchNorm2d(16)(result)
        result = nn.LeakyReLU()(result)
        result = self.encode3(result)
        result = nn.BatchNorm2d(32)(result)
        result = nn.LeakyReLU()(result)
        # print(result.shape)
        #result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # print(result.shape)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        # print("DECODING")
        result = self.decoder_input(z)
        # print(result.shape)
        
        #result = result.view(-1, 512, 2, 2)
        result = result.view(-1, 32, 11, 11)
        result = self.decode1(result)
        result = nn.BatchNorm2d(16)(result)
        result = nn.LeakyReLU()(result)
        # print(result.shape)
        result = self.decode2(result)
        result = nn.BatchNorm2d(8)(result)
        result = nn.LeakyReLU()(result)
        # print(result.shape)
        result = self.decode3(result)
        result = nn.BatchNorm2d(8)(result)
        result = nn.LeakyReLU()(result)
        # print(result.shape)
        result = self.decode4(result)
        result = nn.Tanh()(result)
        # print(result.shape)
        #print(result.shape)
        #result = self.decoder(result)
        #print(result.shape)
        #result = self.final_layer(result)
        #print(result.shape)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        #kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        kld_weight = 1.0
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'recon':recons_loss.detach(), 'KLD':-kld_loss.detach()}



    def sample(self,
               num_samples,
               current_device, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
