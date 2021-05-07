# taken from https://github.com/hypojorik/Machine_Learning/blob/310c7d6374093f3238cdc476c05a1d56f29fb161/research_data_generation/DCGAN.py

import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from numpy.random import choice
from numpy import load
import warnings
warnings.filterwarnings('ignore')
import sys, os, glob, time, imageio
from numpy import asarray
from numpy import savez_compressed
from keras import models, layers, optimizers

IMG_SIZE = 128
# Time
def _time(start, end):
    # if in seconds
    if (end-start)<60:
        wall_time = f'{round((end-start),2)}sec'
    # if in minute(s)
    elif (end-start)>=3600:
        wall_time = f'{int((end-start)/3600)}h {int(((end-start)%3600)/60)}min {round((end-start)%60,2)}sec'
    # if in houre(s)
    else:
        wall_time = f'{int((end-start)/60)}min {round((end-start)%60,2)}sec'
    return wall_time

def list_images(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=(".jpg", ".jpeg", ".png", ".bmp"), contains=contains)

def list_files(basePath, validExts=(".jpg", ".jpeg", ".png", ".bmp"), contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue
            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename).replace(" ", "\\ ")
                yield imagePath


def load_images(directory='', size=(128, 128)):
    images = []
    labels = []  # Integers corresponding to the categories in alphabetical order
    label = 0
    imagePaths = list(list_images(directory))

    for path in imagePaths:
        if not ('OSX' in path):
            path = path.replace('\\', '/')
            image = cv2.imread(path)  # Reading the image with OpenCV
            image = cv2.resize(image, size)  # Resizing the image, in case some are not of the same size
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return images

X_normal= np.array(load_images('Pneumonia_NORMAL',size=(IMG_SIZE, IMG_SIZE)))  # to specify the target size of image
X_normal = (X_normal.astype(np.float32) - 127.5) / 127.5

# Number of training epochs
n_epoch = 500
# Batch size during training
batch_size = 128
# Size of z latent vector (i.e. size of generator input)
latent_dim = 100
# Spatial size of training images. All images will be resized to this size

# Number of channels in the training images. For RGB color images this is 3
channels = 3
in_shape = (IMG_SIZE, IMG_SIZE, channels)  # height, width, color

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# plot ncols images in row and nrows images in colomn
nrows, ncols = 3, 4


def define_discriminator(in_shape=(128, 128, 3)):
    model = models.Sequential()
    # normal
    model.add(layers.Conv2D(64, (5, 5), padding='same', input_shape=in_shape))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample to 64x64
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample to 32x32
    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample to 16x16
    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample to 8x8
    model.add(layers.Conv2D(1024, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # classifier
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1, activation='sigmoid'))
    # compile model
    opt = optimizers.Adam(lr=0.00002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def define_generator(latent_dim):
    model = models.Sequential()
    # foundation for 8x8 feature maps
    n_nodes = 1024 * 8 * 8
    model.add(layers.Dense(n_nodes, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((8, 8, 1024)))
    # upsample to 16x16
    model.add(layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # upsample to 32x32
    model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # upsample to 64x64
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # upsample to 128x128
    model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # output layer 128x128x3
    model.add(layers.Conv2D(3, (5, 5), activation='tanh', padding='same'))
    return model

# input of G
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = np.random.randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = g_model.predict(x_input)
    # create 'fake' class labels (0)
    y = np.zeros((n_samples, 1))
    return X, y


def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect them
    model = models.Sequential()
    # add generator
    model.add(g_model)
    # add the discriminator
    model.add(d_model)
    # compile model
    opt = optimizers.Adam(lr=0.0001, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


# retrive real samples
def get_real_samples(dataset, n_samples):
    # choose random instances
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # set 'real' class labels (1)
    y = np.ones((n_samples, 1))
    return X, y


# create and save a plot of generated images
def show_generated(img_save_dir, generated, epoch, nrows=4, ncols=5):
    # [-1,1] -> [0,1]
    generated = (generated + 1) / 2
    # generated = (generated[:ncols*nrows]*127.5)+127.5
    # generated = generated*255
    plt.figure(figsize=(10, 10))
    for idx in range(nrows * ncols):
        plt.subplot(nrows, ncols, idx + 1)
        plt.imshow(generated[idx])
        plt.axis('off')
    plt.savefig(os.path.join(img_save_dir, 'image_at_epoch_{:04d}.png').format(epoch + 1) )
    plt.show()


# evaluate the discriminator and plot generated images
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
    # prepare real samples
    X_real, y_real = get_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('> Accuracy at epoch %d [real: %.0f%%, fake: %.0f%%]' % (epoch + 1, acc_real * 100, acc_fake * 100))
    # show plot
    # save the generator model tile file
    filename = 'model/generator_model_%03d.h5' % (epoch + 1)
    g_model.save(filename)


def plot_loss(loss):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training", fontsize=20)
    plt.plot(loss[0], label="D_real")
    plt.plot(loss[1], label="D_fake")
    plt.plot(loss[2], label="G")
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.legend()
    plt.show()
def plot_history_loss(dict_his):
    plt.figure(figsize=(10, 5))
    plt.title("Loss history During Training", fontsize=20)
    plt.plot(dict_his["D_real_loss"], label="D_real_loss")
    plt.plot(dict_his["D_fake_loss"], label="D_fake_loss")
    plt.plot(dict_his["D_G_loss"], label="G_loss")
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.legend()
    plt.show()

def plot_history_acc(dict_his):
    plt.figure(figsize=(10, 5))
    plt.title("Acc history During Training", fontsize=20)
    plt.plot(dict_his["D_real_acc"], label="D_real")
    plt.plot(dict_his["D_fake_acc"], label="D_fake")
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Acc", fontsize=20)
    plt.legend()
    plt.show()


def train(g_model, d_model, gan_model, dataset, latent_dim=100, n_epochs=500, n_batch=128, save_model_interval=20, save_img_interval=10, history_interval = 5):
    subs = os.listdir("save_DCGAN")
    new_subdir = os.path.join("save_DCGAN", str(len(subs) + 1))
    os.makedirs(new_subdir, exist_ok=False)
    new_img_dir = os.path.join(new_subdir, "image")
    new_weight_dir = os.path.join(new_subdir, "weight")
    os.makedirs(new_img_dir)
    os.makedirs(new_weight_dir)

    start = time.time()
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    loss1, loss2, loss3, acc1, acc2 = [], [], [], [], []
    dict_loss = {"D_real_loss": [], "D_fake_loss": [], "D_G_loss": []}
    dict_acc = {"D_real_acc": [], "D_fake_acc": []}
    fake_liste = []

    # manually enumerate epochs
    print('Training Start...')
    for i in range(n_epochs):
        start1 = time.time()
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = get_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, d_acc1 = d_model.train_on_batch(X_real, y_real)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, d_acc2 = d_model.train_on_batch(X_fake, y_fake)
            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = np.ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch

            dict_loss["D_real_loss"].append(d_loss1)
            dict_loss["D_fake_loss"].append(d_loss2)
            dict_loss["D_G_loss"].append(g_loss)
            dict_acc["D_real_acc"].append(d_acc1)
            dict_acc["D_fake_acc"].append(d_acc2)


        print('Epoch: {:03d}/{:03d}, Loss: [D_loss_real = {:2.3f}, D_loss_fake = {:2.3f}, G = {:2.3f}], time: {:s}' \
              .format(i + 1, n_epochs, d_loss1, d_loss2, g_loss, _time(start1, time.time())))
        if (i+1) % save_img_interval == 0:
            x_fake, y_fake = generate_fake_samples(g_model, latent_dim, 100)
            show_generated(new_img_dir,generated=x_fake, epoch=i)
        if (i+1) % save_model_interval == 0:
            filename = os.path.join(new_weight_dir,  'generator_model_%03d.h5' % (i + 1))
            g_model.save(filename)
        if (i+1) % history_interval == 0:
            plot_history_loss(dict_loss)
            plot_history_acc(dict_acc)

    print('Total time for training {} epochs is {} sec'.format(n_epochs, _time(start, time.time())))



discriminator = define_discriminator()
generator = define_generator(latent_dim)

# create the gan
gan = define_gan(generator, discriminator)

# train model
train(generator, discriminator, gan, X_normal, latent_dim, n_epochs=n_epoch, n_batch=batch_size,save_model_interval=20, save_img_interval=5)
