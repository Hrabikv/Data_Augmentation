from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras import Sequential
from keras.layers import Flatten


# function for building the discriminator layers
def build_discriminator(img_shape):

    model = Sequential(name="disriminator")

    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))

   # model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)
