import keras.models
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from generator import build_generator_mk_2
from discriminator import build_discriminator


def print_graph(data):
    print(data.shape)


def up_scale(number):
    return number * 100


def down_scale(number):
    return number / 100


class GAN:
    def __init__(self):
        self.vector_size = 225  # Size of random vector
        self.img_rows = 3  # Number of channels of input data
        self.img_cols = 1200  # number of values in one channel
        self.img_shape = (self.img_rows, self.img_cols)  # Shape of one signal in ínput data

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = build_discriminator(self.img_shape)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Build and compile the generator
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
    def predict(self, data_len, percentage=200):
        number_of_new = (data_len * percentage / 100) - data_len
        noise = np.random.normal(0, 1, (int(number_of_new), self.vector_size))

        gen_data = self.generator.predict(noise)
        gen_data = np.array(gen_data)
        print(gen_data.shape)
        return gen_data

    # Function which train GAN for number of epochs
    def train(self, epochs, dataset, name, batch_size=64, save_interval=50):

        for i in range(len(dataset)):
            for j in range(len(dataset[0])):
                for k in range(len(dataset[0][0])):
                    dataset[i][j][k] = down_scale(dataset[i][j][k])

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
                self.save_data_img(epoch, name)

    # Function for print image of predicted signals
    def save_data_img(self, epoch, name):
        rows = 3
        noise = np.random.normal(0, 1, (rows, self.vector_size))
        gen_images = self.generator.predict(noise)

        for i in range(len(gen_images)):
            for j in range(len(gen_images[0])):
                for k in range(len(gen_images[0][0])):
                    gen_images[i][j][k] = up_scale(gen_images[i][j][k])

        for i in range(rows):
            fig, axs = plt.subplots(1, rows)

            for j in range(rows):
                axs[j].plot(gen_images[i][j])

            fig.savefig("training/{0}/P300_{1}_{2}.png".format(name, epoch, i))
        plt.close()

    # Function for saving the GAN
    def save_model(self, name):
        self.generator.save(name)

    # Function for saving the GAN
    def load_model(self, name):
        self.generator = keras.models.load_model(name)
