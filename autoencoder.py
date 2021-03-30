import tensorflow as tf


class Autoencoder(tf.keras.Model):

    def __init__(self, name="autoencoder", **kwargs):
        super(Autoencoder, self).__init__(name=name, **kwargs)

    def call(self, inputs, training=None, mask=None):
        return

    def get_config(self):
        raise NotImplementedError("Not implemented")


class Encoder(tf.keras.Model):

    def __init__(self, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

    def call(self, inputs, training=None, mask=None):
        return

    def get_config(self):
        raise NotImplementedError("Not implemented")


class Decoder(tf.keras.Model):

    def __init__(self, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

    def call(self, inputs, training=None, mask=None):
        return

    def get_config(self):
        raise NotImplementedError("Not implemented")
