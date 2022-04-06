import numpy as np
from keras.layers import Reshape, Input, Dense, BatchNormalization
from keras.models import Model
from keras import Sequential
from keras.layers.advanced_activations import LeakyReLU


# function for building the generator layers of first model
def build_generator_mk_1(img_shape, gan):
    noise_shape = (gan.vector_size,)

    model = Sequential(name="generator_mk_1")

    model.add(Dense(1200, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(2400))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))

    # model.summary()

    noise = Input(shape=noise_shape)
    img = model(noise)
    return Model(noise, img)


# function for building the generator layers of second model
def build_generator_mk_2(img_shape, gan):
    noise_shape = (gan.vector_size,)

    model = Sequential(name="generator_mk_2")

    model.add(Dense(450, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(900))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1800))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))

    # model.summary()

    noise = Input(shape=noise_shape)
    img = model(noise)
    return Model(noise, img)
