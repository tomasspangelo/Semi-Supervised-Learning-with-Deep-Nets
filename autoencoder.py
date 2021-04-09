import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, Dense, Reshape


class Autoencoder(tf.keras.Model):
    """
    Class for Autoencoder. Inherits from Keras' Model class.
    """

    def __init__(self, image_shape, latent_size, name="autoencoder", **kwargs):
        """
        Initializes variables and calls constructor of superclass.
        :param image_shape: Shape of the input images.
        :param latent_size: Size of the latent vector.
        :param name: Name of the model.
        :param kwargs: Other arguments which are appropriate for the Model class.
        """
        super(Autoencoder, self).__init__(name=name, **kwargs)
        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(image_shape)
        self.latent_size = latent_size

    def call(self, inputs, training=None, mask=None):
        """
        Forward pass for the autoencoder.
        :param inputs: Input data.
        :param training: N/A
        :param mask: N/A
        :return: Output from model.
        """
        x = self.encoder(inputs)
        return self.decoder(x)

    def get_config(self):
        """ Method inherited from superclass."""
        raise NotImplementedError("Not implemented")


class Encoder(tf.keras.Model):
    """
    Class for Encoder. Inherits from Keras' Model class.
    """

    def __init__(self, latent_size, name="encoder", **kwargs):
        """
        Initializes variables and calls constructor of superclass.
        :param latent_size: Size of the latent vector.
        :param name: Name of the model.
        :param kwargs: Other arguments which are appropriate for the Model class.
        """
        super(Encoder, self).__init__(name=name, **kwargs)
        self.conv1 = Conv2D(filters=16, kernel_size=3, activation='relu', padding="same", strides=2)
        self.conv2 = Conv2D(filters=8, kernel_size=3, activation='relu', padding="same", strides=2)
        self.conv3 = Conv2D(filters=4, kernel_size=3, activation='relu', padding="same")
        self.flatten = Flatten()
        self.latent = Dense(units=latent_size, activation='tanh')

    def call(self, inputs, training=None, mask=None):
        """
        Forward pass for the encoder.
        :param inputs: Input data.
        :param training: N/A
        :param mask: N/A
        :return: Output from model.
        """
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        return self.latent(x)

    def get_config(self):
        """ Method inherited from superclass."""
        raise NotImplementedError("Not implemented")

    def freeze_model(self):
        """Freezes the model so that the layers are not trainable."""
        self.conv1.trainable = False
        self.conv2.trainable = False
        self.conv3.trainable = False
        self.flatten.trainable = False
        self.latent.trainable = False

    def unfreeze_model(self):
        """Unfreezes the model so that the layers are trainable."""
        self.conv1.trainable = True
        self.conv2.trainable = True
        self.conv3.trainable = True
        self.flatten.trainable = True
        self.latent.trainable = True


class Decoder(tf.keras.Model):
    """
    Class for Decoder. Inherits from Keras' Model class.
    """

    def __init__(self, image_shape, name="decoder", **kwargs):
        """
        Initializes variables and calls constructor of superclass.
        :param image_shape: Shape of the input image.
        :param name: Name of the model.
        :param kwargs: Other arguments which are appropriate for the Model class.
        """
        super(Decoder, self).__init__(name=name, **kwargs)
        height = int(image_shape[0] / 4)
        width = int(image_shape[1] / 4)
        self.dense = Dense(units=height * width * 4, activation='tanh')
        self.reshape = Reshape((height, width, 4))
        self.conv1 = Conv2DTranspose(filters=4, kernel_size=3, activation="relu", padding="same")
        self.conv2 = Conv2DTranspose(filters=8, kernel_size=3, activation="relu", padding="same", strides=2)
        self.conv3 = Conv2DTranspose(filters=16, kernel_size=3, activation="relu", padding="same", strides=2)
        self.conv4 = Conv2D(filters=1, kernel_size=(3, 3), activation="sigmoid", padding="same")

    def call(self, inputs, training=None, mask=None):
        """
        Forward pass for the decoder.
        :param inputs: Input data.
        :param training: N/A
        :param mask: N/A
        :return: Output from model.
        """
        x = self.dense(inputs)
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.conv4(x)

    def get_config(self):
        """ Method inherited from superclass."""
        raise NotImplementedError("Not implemented")


if __name__ == "__main__":
    pass
