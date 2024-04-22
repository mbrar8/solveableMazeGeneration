import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch
from dataset import VAEDataset
from models import VanillaVAE
from torchvision.transforms import ToTensor
# from solveableMazeGeneration.maze_generator_dataset import MazeGeneratorDataset
from solveableMazeGeneration.maze_dataset import MazeDataset


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
from torch.utils.data import DataLoader, random_split
# data.setup()
print(f"======= Training =======")
input_image_path = "../mazes/saved_imgs100"
input_channels = 3
latent_dim = 256
hidden_dims = [32, 64, 128, 256, 512]
vae = VanillaVAE(input_channels, latent_dim, hidden_dims)
num_epochs = 1000
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
dataset = MazeDataset(transform=ToTensor())
train_set, val_set = random_split(dataset, [0.8, 0.2])
train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=32, shuffle=True)
for epoch in range(num_epochs):
    for image, _ in train_dataloader:
        optimizer.zero_grad()
        # image_path = os.path.join(input_image_path, image)

        recon_imgs, inputs, mu, log_var = vae.forward(image)
        loss = vae.loss_function(recon_imgs, inputs, mu, log_var)
        loss.backward()
        optimizer.step()

print("""======= Training Finished =======""")






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