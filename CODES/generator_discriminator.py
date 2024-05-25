from keras.layers import *
from keras.models import Model
from keras.utils import plot_model


# class for creating discriminator and generator models
class Models:
    """This method creates the models"""

    def __init__(self):
        self.latent_dim = 128

    # discriminator
    @staticmethod
    def discriminator():
        # discriminator model

        input_layer = Input(shape=(128, 128, 3))

        # conv layer
        conv1 = Conv2D(64, 3, 2, "same")(input_layer)
        leakyrelu1 = LeakyReLU(0.2)(conv1)

        layer_norm1 = LayerNormalization()(leakyrelu1)

        # conv layer
        conv2 = Conv2D(128, 3, 2, "same")(layer_norm1)
        leakyrelu2 = LeakyReLU(0.2)(conv2)
        dropout2 = Dropout(0.2)(leakyrelu2)

        # layer norm
        layer_norm2 = LayerNormalization()(dropout2)

        # conv layer
        conv3 = Conv2D(256, 3, 2, "same")(layer_norm2)
        leakyrelu3 = LeakyReLU(0.2)(conv3)
        dropout3 = Dropout(0.2)(leakyrelu3)

        # layer norm 3
        layer_norm3 = LayerNormalization()(dropout3)

        # conv layer
        conv4 = Conv2D(512, 3, 2, "same")(layer_norm3)
        leakyrelu4 = LeakyReLU(0.2)(conv4)
        dropout4 = Dropout(0.2)(leakyrelu4)

        # layer norm 4
        layer_norm4 = LayerNormalization()(dropout4)

        # flatten
        flatten = Flatten()(layer_norm4)

        # dense without sigmoid
        last_dense = Dense(1)(flatten)

        discriminator = Model(input_layer, last_dense, name="discriminator")

        # plot model
        plot_model(model=discriminator, to_file="discriminator.png")

        # return discriminator
        return discriminator

    # generator
    def generator(self):
        # input
        input_layer1 = Input(shape=(self.latent_dim,))

        # dense
        input_layer2 = Dense(8 * 8 * 512, use_bias=False)(input_layer1)
        input_layer3 = Reshape((8, 8, 512))(input_layer2)

        # conv_t_1
        conv_t_1 = Conv2DTranspose(128, 3, 2, "same", use_bias=False)(input_layer3)
        bn = BatchNormalization()(conv_t_1)
        leakyrelu1 = LeakyReLU(0.2)(bn)

        # conv_t_2
        conv_t_2 = Conv2DTranspose(256, 3, 2, "same", use_bias=False)(leakyrelu1)
        bn2 = BatchNormalization()(conv_t_2)
        leakyrelu2 = LeakyReLU(0.2)(bn2)

        # conv_t_3
        conv_t_3 = Conv2DTranspose(512, 3, 2, "same", use_bias=False)(leakyrelu2)
        bn3 = BatchNormalization()(conv_t_3)
        leakyrelu3 = LeakyReLU(0.2)(bn3)

        # conv_t_4
        conv_t_4 = Conv2DTranspose(256, 3, 2, "same", use_bias=False)(leakyrelu3)
        bn4 = BatchNormalization()(conv_t_4)
        leakyrelu4 = LeakyReLU(0.2)(bn4)


        # conv
        to_rgb_result = Conv2D(3, 3, 1, "same", use_bias=False, activation="tanh")(leakyrelu4)

        generator = Model(input_layer1, to_rgb_result, name="generator")

        # plot the generator

        plot_model(model=generator, to_file="generator.png")

        return generator


create_models = Models()
