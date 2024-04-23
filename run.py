import os
import yaml
import argparse
import numpy as np
from pathlib import Path
#from models import *
#from experiment import VAEXperiment
import torch
#from dataset import VAEDataset
#from models import VanillaVAE
from torchvision.transforms import ToTensor
# from solveableMazeGeneration.maze_generator_dataset import MazeGeneratorDataset
from maze_dataset import MazeDataset
from vanilla_vae import VanillaVAE
from torch.utils.data import DataLoader, random_split

# parser = argparse.ArgumentParser(description='Generic runner for VAE models')
# parser.add_argument('--config',  '-c',
#                     dest="filename",
#                     metavar='FILE',
#                     help =  'path to the config file',
#                     default='configs/vae.yaml')

# args = parser.parse_args()
# with open(args.filename, 'r') as file:
#     try:
#         config = yaml.safe_load(file)
#     except yaml.YAMLError as exc:
#         print(exc)


# tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
#                                name=config['model_params']['name'],)

# # For reproducibility
# seed_everything(config['exp_params']['manual_seed'], True)

# model = vae_models[config['model_params']['name']](**config['model_params'])
# experiment = VAEXperiment(model,
                        #   config['exp_params'])

# data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)

# data.setup()
generator = torch.Generator()
generator.manual_seed(0)
input_channels = 3
latent_dim = 256
hidden_dims = [32, 64, 128, 256, 512]
num_epochs = 100
vae = VanillaVAE(input_channels, latent_dim, hidden_dims)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
dataset = MazeDataset(transform=ToTensor())
train_set, val_set = random_split(dataset, [0.8, 0.2], generator=generator)
train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=32, shuffle=True)

def train():
    print(f"======= Training =======")
    vae.train()
    for epoch in range(num_epochs):
        print("                                                                            Epoch " + str(epoch))
        for image, _ in train_dataloader:
            vae.train()
            optimizer.zero_grad()
            # image_path = os.path.join(input_image_path, image)

            recon_imgs, inputs, mu, log_var = vae.forward(image)
            losses  = vae.loss_function(recon_imgs, inputs, mu, log_var)
            loss = losses['loss']
            recon_loss = losses['recon']
            kld = losses['KLD']
            print("LOSS: " + str(loss))
            print("Reconstruction Loss: " + str(recon_loss))
            print("KLD: " + str(kld))
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            validate()

    print("""======= Training Finished =======""")
    torch.save(vae.state_dict(), 'vae.pth')

def validate():
    print(f"======= Validation =======")
    vae.eval()
    for image, _ in val_dataloader:
        recon_img, inputs, mu, log_var = vae.forward(image)
        val_loss = vae.loss_function(recon_img, inputs, mu, log_var)

        loss = val_loss['loss']
        recon_loss = val_loss['recon']
        kld = val_loss['KLD']
        print("VALIDATION LOSS: " + str(loss))
        print("VALIDATION Reconstruction Loss: " + str(recon_loss))
        print("VALIDATION KLD: " + str(kld))


if __name__ == "__main__":
    train()
    vae = torch.load('vae.pth')




# runner = Trainer(logger=tb_logger,
#                  callbacks=[
#                      LearningRateMonitor(),
#                      ModelCheckpoint(save_top_k=2, 
#                                      dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
#                                      monitor= "val_loss",
#                                      save_last= True),
#                  ],
#                  strategy=DDPPlugin(find_unused_parameters=False),
#                  **config['trainer_params'])


# Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
# Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)



# runner.fit(experiment, datamodule=data)
