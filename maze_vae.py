# https://github.com/antigones/pymazes-vae/blob/main/maze_vae.py

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import os
from sampling import Sampling
from image_utils import read_images, sample_to_image
import tensorflow.keras.backend as K


def plot_latent_space(decoder, maze_size=28, n=30, scale=15):
    # display a n*n 2D manifold of mazes
    figure = np.zeros((maze_size * n, maze_size * n))
    # We will sample n points within [-scale, scale] standard deviations
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            maze = x_decoded[0].reshape(maze_size, maze_size)
            figure[i * maze_size: (i + 1) * maze_size, j * maze_size: (j + 1) * maze_size] = maze

    plt.figure(figsize=(maze_size, maze_size))
    plt.imshow(figure)
    plt.show()


def plot_history(history):
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def get_prediction(decoder, maze_size, n=30, scale=1.0):

    #  get a point in latent space, to be decoded
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)

    xi = np.random.choice(grid_x)
    yi = np.random.choice(grid_y)

    z_sample = np.array([[xi, yi]])
    x_decoded = decoder.predict(z_sample)
    # breakpoint()
    maze = x_decoded[0].reshape(108, 108, 3)
    return maze


def main():

    #  build the encoder
    size = 36*3  # maze edge * 3
    original_dim = size * size * 3
    intermediate_dim = 64
    latent_dim = 2

    inputs = keras.Input(shape=(original_dim))
    x = layers.Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    #  build the decoder
    latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
    x = layers.Dense(intermediate_dim, activation="relu")(latent_inputs)
    outputs = layers.Dense(original_dim, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs, name="decoder")
    decoder.summary()

    #  train the VAE
    x = read_images("./../mazes/saved_imgs100/")
    mask = read_images("./../mazes/mask_imgs100/")

    x_train, x_test = keras.utils.split_dataset(x, left_size=0.8, seed=42)
    mask_train, mask_test = keras.utils.split_dataset(mask, left_size=0.8, seed=42)
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.concatenate((x_train, mask_train), axis=1)
    x_test = np.concatenate((x_test, mask_test), axis=1)
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    outputs = decoder(encoder(inputs)[2])
    vae = keras.Model(inputs, outputs, name='vae_mlp')

    #  re-adapted to use tf from https://blog.keras.io/building-autoencoders-in-keras.html
    reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss *= original_dim
    kl_loss = (1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer=keras.optimizers.Adam())

    checkpoint_filepath = os.curdir + '/checkpoints'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='auto',
        save_best_only=True)

    history = vae.fit(
        x_train,
        x_train,
        epochs=1000,
        batch_size=128,
        callbacks=[model_checkpoint_callback, tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss')],
        validation_data=(x_test, x_test))
    plot_history(history)

    vae.load_weights(checkpoint_filepath)
    scale = 100
    # plot_latent_space(decoder, maze_size = size, n=8, scale=scale)

    pred = get_prediction(decoder, maze_size=size, n=8, scale=scale)
    sample_to_image(pred, "output.jpg")


if __name__ == '__main__':
    main()