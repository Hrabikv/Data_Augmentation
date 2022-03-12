import numpy as np
from keras.layers import Reshape, BatchNormalization, Input, Dense
from keras.models import Model
from keras import Sequential
from keras.layers.advanced_activations import LeakyReLU


def build_generator(img_shape):
    noise_shape = (1000,)

    model = Sequential(name="generator")

    model.add(Dense(1200, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(3600))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))

    model.summary()

    noise = Input(shape=noise_shape)
    img = model(noise)
    return Model(noise, img)
