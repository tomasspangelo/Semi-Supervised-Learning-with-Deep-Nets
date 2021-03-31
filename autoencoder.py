import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, Dense, Reshape
from tensorflow.keras.datasets import mnist, fashion_mnist


class Autoencoder(tf.keras.Model):

    def __init__(self, image_shape, name="autoencoder", **kwargs):
        super(Autoencoder, self).__init__(name=name, **kwargs)
        self.encoder = Encoder(image_shape)
        self.decoder = Decoder(image_shape)

    def call(self, inputs, training=None, mask=None):
        x = self.encoder(inputs)
        return self.decoder(x)

    def get_config(self):
        raise NotImplementedError("Not implemented")


class Encoder(tf.keras.Model):

    def __init__(self, image_shape, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.conv1 = Conv2D(filters=16, kernel_size=3, activation='relu', padding="same", strides=2)
        self.conv2 = Conv2D(filters=8, kernel_size=3, activation='relu', padding="same", strides=2)
        self.conv3 = Conv2D(filters=4, kernel_size=3, activation='relu', padding="same")
        self.flatten = Flatten()
        height = int(image_shape[0] / 4)
        width = int(image_shape[1] / 4)
        self.latent = Dense(units=height*width*4, activation='tanh')

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        print(x.shape)
        x = self.flatten(x)
        return self.latent(x)

    def get_config(self):
        raise NotImplementedError("Not implemented")

    def freeze_model(self):
        self.conv1.trainable = False
        self.conv2.trainable = False
        self.conv3.trainable = False
        self.flatten.trainable = False
        self.latent.trainable = False

    def unfreeze_model(self):
        self.conv1.trainable = True
        self.conv2.trainable = True
        self.conv3.trainable = True
        self.flatten.trainable = True
        self.latent.trainable = True


class Decoder(tf.keras.Model):

    def __init__(self, image_shape, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        height = int(image_shape[0]/4)
        width = int(image_shape[1]/4)
        self.reshape = Reshape((height, width, 4))
        self.conv1 = Conv2DTranspose(filters=4, kernel_size=3, activation="relu", padding="same")
        self.conv2 = Conv2DTranspose(filters=8, kernel_size=3, activation="relu", padding="same", strides=2)
        self.conv3 = Conv2DTranspose(filters=16, kernel_size=3, activation="relu", padding="same", strides=2)
        self.conv4 = Conv2D(filters=1, kernel_size=(3,3), activation="tanh", padding="same")

    def call(self, inputs, training=None, mask=None):
        x = self.reshape(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.conv4(x)

    def get_config(self):
        raise NotImplementedError("Not implemented")


if __name__ == "__main__":
    #(x_train, y_train), (x_test, y_test) = mnist.load_data(path="mnist.npz")
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.reshape(x_train.shape + (1,))[:100]

    #x_train = tf.image.decode_jpeg(x_train)
    x_train = tf.cast(x_train, tf.float32)/255.0
    print(x_train.shape)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.MeanSquaredError()
    ae = Autoencoder(image_shape=(28, 28))
    ae.compile(optimizer=optimizer, loss=loss)
    ae.fit(x_train, x_train, epochs=200, batch_size=5)



