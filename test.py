import numpy as np
import torch
from vanilla_vae import VanillaVAE
import matplotlib.pyplot as plt


vae = torch.load('vae.pth')
vae.eval()


mazes = vae.sample(10, 'cpu').numpy()
i = 0
for maze in mazes:
    plt.imshow(maze)
    plt.axis('off')
    plt.savefig("maze_example_" + str(i) + ".png")
    i += 1


