

import numpy as np
import os
import cv2
import glob
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, BatchNormalization, MaxPooling2D, Dropout, Dense , Activation, GlobalAveragePooling2D
from keras.layers import AveragePooling2D
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Concatenate
import matplotlib.pyplot as plt

pneunomial_classes = ["NORMAL", "PNEUMONIA"]
NUM_CLASSES = len(pneunomial_classes)
IMG_SIZE = 128

def load_data(target_size = (IMG_SIZE,IMG_SIZE), img_folder = ""):
    X = []
    Y = []
    for pneumonial_type in pneunomial_classes:
        imgdir = os.path.join(img_folder, pneumonial_type)
        imgnames = os.listdir(imgdir)
        for imgname in imgnames:
            imgpath = os.path.join(imgdir, imgname)
            img = cv2.imread(imgpath)
            img = cv2.resize(img, target_size)
            img = (np.array(img).astype('float32') - 127.5)/127.5
            X.append(img)
            Y.append(pneunomial_classes.index(pneumonial_type))
    return [np.array(X), np.array(Y)]

def discriminator(in_shape = (IMG_SIZE,IMG_SIZE,3), n_classes = NUM_CLASSES):
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(n_classes, 50)(in_label)
    # scale up to image dimensions with linear activation
    n_nodes = in_shape[0]*in_shape[1]
    li = Dense(n_nodes)(li)
    # reshape to additional channel
    li = Reshape((in_shape[0], in_shape[1],1))(li)
    # image input
    in_image = Input(shape=in_shape)
    # concat label as a channel
    merge = Concatenate()([in_image, li]) # merge.shape = (128, 128, 4)

    # normal
    x = Conv2D(64, (5,5), padding='same')(merge)
    x = LeakyReLU(alpha=0.2)(x)

    # downsample to 64x64
    x = Conv2D(128, (5,5), strides=(2,2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)

    # downsample to 32x32
    x = Conv2D(256, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)

    # downsample to 16x16
    x = Conv2D(512, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)

    # downsample to 8x8
    x = Conv2D(1024, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)

    # classifier
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[in_image, in_label], outputs=x)

    # compile model
    opt = Adam(lr=0.00002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def generator(latent_dim = 100, n_classes = NUM_CLASSES):
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(n_classes, 50)(in_label)
    n_nodes = 8*8
    li = Dense(n_nodes)(li)
    # reshape to additional channel
    li = Reshape((8,8,1))(li)
    # image generator input
    in_lat = Input(shape=(latent_dim,))
    # foundation for 6*6 image
    n_nodes = 1023*8*8
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((8,8,1023))(gen)
    # merge image gen and label
    merge = Concatenate()([gen, li]) # merge.shape = (8, 8, 1024)

    # upsample to 16x16
    x = Conv2DTranspose(filters=512, kernel_size=(4,4), strides=(2,2), padding='same')(merge)
    x = LeakyReLU(alpha=0.2)(x)

    # upsample to 32x32
    x = Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)

    # upsample to 64x64
    x = Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)

    # upsample to 128x128
    x = Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)

    # output layer 128x128x3
    x = Conv2D(filters=3, kernel_size=(5,5), padding='same', activation='tanh')(x)

    model = Model([in_lat, in_label], x)
    return model


def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        layer.trainable = False

    # get noise and label inputs from generator model
    gen_noise, gen_label = g_model.input
    # get image output from the generator model
    gen_output = g_model.output
    # connect image output and label input from generator as inputs to discriminator
    gan_output = d_model([gen_output, gen_label])
    # define gan model as taking noise and label and outputting a classification
    model = Model([gen_noise, gen_label], gan_output)
    # compile model
    opt = Adam(lr=0.0001, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

def generate_real_samples(dataset, n_samples):
    images, labels = dataset
    ix = randint(0, images.shape[0], n_samples)
    X, labels = images[ix], labels[ix]
    y = ones((n_samples, 1))
    return [X, labels], y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=NUM_CLASSES):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict([z_input, labels_input])
    # create class labels
    y = zeros((n_samples, 1))
    return [images, labels_input], y

# functions for recording training history
def save_model(discriminator, generator, gan, save_dir, epoch):
    d_path = os.path.join(save_dir, "discriminator_epoch_" + str(epoch) + ".h5")
    g_path = os.path.join(save_dir, "generator_epoch_" + str(epoch) + ".h5")
    gan_path = os.path.join(save_dir, "gan_epoch_" + str(epoch) + ".h5")
    discriminator.save(d_path)
    generator.save(g_path)
    gan.save(gan_path)

def generate_image(epoch, save_dir, generator, n_classes=NUM_CLASSES, latent_dim = 100):
    n_samples_each_classes = 5
    labels = []
    for i in range(n_classes):
        tmp_ = [i for _ in range(n_samples_each_classes)]
        labels.extend(tmp_)
    labels = np.array(labels)

    # generate points in the latent space
    x_input = randn(latent_dim * n_classes * n_samples_each_classes)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_classes * n_samples_each_classes, latent_dim)

    images = generator.predict([z_input, labels])
    images = (images+1)/2.0

    columns = n_samples_each_classes
    rows = n_classes

    fig = plt.figure(figsize=(20, 8))
    for i in range(1, columns*rows+1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(images[i-1,:,:,0])
    save_name = 'epoch_' + str(epoch) + '.png'
    plt.savefig(os.path.join(save_dir, save_name))
    plt.show()

def plot_history(dict_, save_dir, epoch):
    d_loss_real = dict_["d_loss_real"]
    d_acc_real = dict_["d_acc_real"]
    d_loss_fake = dict_["d_loss_fake"]
    d_acc_fake = dict_["d_acc_fake"]
    g_loss = dict_["g_loss"]

    fig = plt.figure(figsize=(16, 4))
    fig.add_subplot(1, 3, 1)
    plt.plot(d_loss_real, color='red')
    plt.plot(d_loss_fake, color='blue')
    plt.title('d_loss')
    plt.xlabel("Iteration", fontsize=10)
    plt.ylabel("Loss", fontsize=10)

    fig.add_subplot(1, 3, 2)
    plt.plot(d_acc_real, color='red')
    plt.plot(d_acc_fake, color='blue')
    plt.title('d_acc')
    plt.xlabel("Iteration", fontsize=10)
    plt.ylabel("Acc", fontsize=10)

    fig.add_subplot(1, 3, 3)
    plt.plot(g_loss)
    plt.title('g_loss')
    plt.xlabel("Iteration", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(save_dir, "epoch_" + str(epoch) + ".png"))


# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=32, save_img_interval = 5, save_model_interval = 20, history_interval = 5):
    subdirs = os.listdir("save_CGAN")
    new_subdir = os.path.join("save_CGAN", str(len(subdirs) + 1))
    os.makedirs(new_subdir, exist_ok=False)
    os.makedirs(os.path.join(new_subdir, "history"))
    os.makedirs(os.path.join(new_subdir, "weight"))
    os.makedirs(os.path.join(new_subdir, "image"))

    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    dict_history = {"d_loss_real": [], "d_acc_real":[], "d_loss_fake":[], "d_acc_fake":[], "g_loss":[]}
    for i in range(n_epochs):
        # enumerate batches over the training set
        tmp_dict = {"d_loss_real": [], "d_acc_real":[], "d_loss_fake":[], "d_acc_fake":[], "g_loss":[]}
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights with real samples
            d_loss_real, d_acc_real = d_model.train_on_batch([X_real, labels_real], y_real)
            # generate 'fake' examples
            [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights with fake samples
            d_loss_fake, d_acc_fake = d_model.train_on_batch([X_fake, labels], y_fake)
            # prepare points in latent space as input for the generator
            [z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)

            dict_history["d_loss_real"].append(d_loss_real)
            dict_history["d_acc_real"].append(d_acc_real)
            dict_history["d_loss_fake"].append(d_loss_fake)
            dict_history["d_acc_fake"].append(d_acc_fake)
            dict_history["g_loss"].append(g_loss)

        print("epoch ", i, ": ", "d_loss_real: ", d_loss_real, ", d_acc_real: ", d_acc_real, ", d_loss_fake: ", d_loss_fake, ", d_acc_fake: ", d_acc_fake, ", g_loss: ", g_loss)
        if i % save_img_interval == 0:
            generate_image(epoch=i, save_dir=os.path.join(new_subdir, "image"), generator=g_model, latent_dim=latent_dim)

        if i % save_model_interval == 0:
            save_model(discriminator=d_model, generator=g_model, gan=gan_model, save_dir=os.path.join(new_subdir, "weight"), epoch=i)

        if i % history_interval == 0:
            plot_history(dict_=dict_history, save_dir = os.path.join(new_subdir, "history"), epoch=i)


n_epoch = 500
batch_size = 128
latent_dim = 100

d_model = discriminator()
g_model = generator()
gan_model = define_gan(g_model, d_model)
dataset = load_data(target_size=(IMG_SIZE,IMG_SIZE), img_folder="Pneumonia_condition")
train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs= n_epoch, n_batch= batch_size)


