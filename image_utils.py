import imageio
from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt


def read_images(subdir: str):

    """
    reads images from a given subdir
    """

    imgs_subdir = os.curdir + subdir
    out = []
    for filename in os.listdir(imgs_subdir):
        with open(imgs_subdir+filename, 'r'):
            img_path = os.path.join(imgs_subdir, filename)
            img = np.array(Image.open(img_path).convert('RGB').crop((145, 60, 510, 425)).resize((108, 108)))
            # img = imageio.imread(imgs_subdir+filename)
            # print(np.asarray(img).shape)
            # breakpoint()
            # out.append(np.asarray(img.crop((145, 60, 510, 425)).reshape(108, 108, 4)))
            out.append(img)
    return np.asarray(out)


def sample_to_image(pred: np.ndarray, img_name: str):

    """
    save generated image
    """

    img_pixels = pred.copy()
    # breakpoint()
    img_pixels = img_pixels * 255
    
    img_pixels[img_pixels > 0.8] = 255
    img_pixels[img_pixels <= 0.8] = 0
    plt.imshow(img_pixels)
    plt.savefig(img_name)
    # im = Image.fromarray(img_pixels)
    # if im.mode != 'RGB':
    #     im = im.convert('RGB')
    # imageio.imsave(img_name, im)