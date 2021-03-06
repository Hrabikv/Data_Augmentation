import os.path

import keras.models
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from generator import build_generator_mk_1, build_generator_mk_2
from discriminator import build_discriminator
from DataWork import merge_data


def print_graph(data):
    print(data.shape)


def up_scale(gen_data):
    for o in range(len(gen_data)):
        for n in range(len(gen_data[0])):
            for k in range(len(gen_data[0][0])):
                gen_data[o][n][k] = gen_data[o][n][k] * 100
    return gen_data


def down_scale(gen_data):
    for o in range(len(gen_data)):
        for n in range(len(gen_data[0])):
            for k in range(len(gen_data[0][0])):
                gen_data[o][n][k] = gen_data[o][n][k] / 100
    return gen_data


def average_of_signals(window, gen_target_data):
    average_data = []
    i = 0
    new_element = []
    for element in gen_target_data:
        if i == 0:
            new_element = element
            i += 1
        else:
            new_element += element
            i += 1
        if i == window:
            average_data.append(new_element / i)
            i = 0

    if i > window / 2:
        average_data.append(new_element / i - 1)
    return np.array(average_data)


class GAN:
    def __init__(self, model):
        self.directory = "training"
        if model == "1":
            self.vector_size = 1000  # Size of random vector
        else:
            self.vector_size = 225  # Size of random vector
        self.img_rows = 3  # Number of channels of input data
        self.img_cols = 1200  # number of values in one channel
        self.img_shape = (self.img_rows, self.img_cols)  # Shape of one signal in input data

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = build_discriminator(self.img_shape)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Build and compile the generator
        if model == "1":
            self.generator = build_generator_mk_1(self.img_shape, self)
        else:
            self.generator = build_generator_mk_2(self.img_shape, self)

        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated images
        z = Input(shape=(self.vector_size,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    # Function to determine new signals
    def predict(self, data_len, percentage=200, window=1):
        number_of_new = (data_len * percentage / 100) - data_len
        noise = np.random.normal(0, 1, (int(number_of_new), self.vector_size))
        merge = []
        for i in range(window):
            gen_data = self.generator.predict(noise)
            gen_data = average_of_signals(window, np.array(gen_data))
            merge = merge_data(merge, gen_data)
            print("{0} from {1}".format(i+1, window))

        merge = up_scale(merge)
        # print(merge.shape)
        return merge

    # Function which train GAN for number of epochs
    def train(self, epochs, dataset, name, examples, batch_size=64, save_interval=50):
        if examples == "T":
            parent_dir = "./" + self.directory
            try:
                path = os.path.join(parent_dir, name)
                os.makedirs(path)
            except OSError:
                print()

        dataset = down_scale(dataset)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, dataset.shape[0], half_batch)
            images = dataset[idx]

            noise = np.random.normal(0, 1, (half_batch, self.vector_size))

            # Generate a half batch of new images
            gen_images = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(images, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_images, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.vector_size))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress

            # If at save interval => save generated image samples

            if epoch % save_interval == 0:
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
                if examples == "True":
                    self.save_data_img(epoch, name)

    # Function for print image of predicted signals
    def save_data_img(self, epoch, name):
        rows = 3
        noise = np.random.normal(0, 1, (rows, self.vector_size))
        gen_images = self.generator.predict(noise)

        gen_images = up_scale(gen_images)

        x = []
        for i in range(-200, 1000):
            x.append(i)

        for i in range(rows):
            fig, axs = plt.subplots(1, rows)

            for j in range(rows):
                axs[j].plot(x, gen_images[i][j])
                axs[j].set_xlabel("ms")
                axs[j].set_xticks(np.arange(-200, 1050, 500))

            axs[0].set_ylabel("uV")
            axs[0].set_title("Fz")
            axs[1].set_title("Cz")
            axs[2].set_title("Pz")
            fig.savefig("{0}/{1}/P300_{2}_{3}.png".format(self.directory, name, epoch, i))
        plt.close()

    # Function for saving the GAN
    def save_model(self, name):
        self.generator.save(name)

    # Function for saving the GAN
    def load_model(self, name):
        self.generator = keras.models.load_model(name)
