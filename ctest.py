import numpy as np
import torch
from vanilla_vae import VanillaVAE
import matplotlib.pyplot as plt


vae = torch.load('vae.pth')
vae.eval()


mazes = vae.sample(10, 'cpu').numpy()
i = 0
for maze in mazes:
    img = maze[:-1]
    mask = maze[-1]
    plt.imshow(img)
    plt.axis('off')
    plt.savefig("maze_example_" + str(i) + ".png")
    plt.clf()
    plt.imshow(mask)
    plt.axis('off')
    plt.savefig("mask_example_" + str(i) + ".png")
    i += 1


