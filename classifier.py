import tensorflow as tf
from tensorflow.keras.layers import Dense, Softmax


class Classifier(tf.keras.Model):
    def __init__(self, encoder, num_classes, name="classifier", **kwargs):
        super(Classifier, self).__init__(name=name, **kwargs)
        self.encoder = encoder
        self.classifier_head = ClassifierHead(num_classes)

    def call(self, inputs, training=None, mask=None):
        x = self.encoder(inputs)
        return self.classifier_head(x)

    def get_config(self):
        raise NotImplementedError("Not implemented")


class ClassifierHead(tf.keras.Model):
    def __init__(self, num_classes, name="classifierhead", **kwargs):
        super(ClassifierHead, self).__init__(name=name, **kwargs)
        self.dense1 = Dense(60, activation="tanh")
        self.dense2 = Dense(num_classes, activation="sigmoid")
        self.softmax = Softmax()

    def call(self, inputs, training=None, mask=None):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.softmax(x)

    def get_config(self):
        raise NotImplementedError("Not implemented")
