import tensorflow as tf
from tensorflow.keras.layers import Dense, Softmax


class Classifier(tf.keras.Model):
    """Class for Classifier. Inherits from Keras' Model class."""
    def __init__(self, encoder, num_classes, name="classifier", **kwargs):
        """
        Initializes variables and calls constructor of superclass.
        :param encoder: The encoder for the classifier, may or may not be pre-trained.
        :param num_classes: Number of different classes for the data.
        :param name: Name of the model.
        :param kwargs: Other arguments which are appropriate for the Model class.
        """
        super(Classifier, self).__init__(name=name, **kwargs)
        self.encoder = encoder
        self.classifier_head = ClassifierHead(num_classes)

    def call(self, inputs, training=None, mask=None):
        """
        Forward pass for the classifier.
        :param inputs: Input data.
        :param training: N/A
        :param mask: N/A
        :return: Output from model.
        """
        x = self.encoder(inputs)
        return self.classifier_head(x)

    def get_config(self):
        """ Method inherited from superclass."""
        raise NotImplementedError("Not implemented")


class ClassifierHead(tf.keras.Model):
    """Class for ClassifierHead. Inherits from Keras' Model class."""
    def __init__(self, num_classes, name="classifierhead", **kwargs):
        """
        Initializes variables and calls constructor of superclass.
        :param num_classes: Number of different classes for the data.
        :param name: Name of the model.
        :param kwargs: Other arguments which are appropriate for the Model class.
        """
        super(ClassifierHead, self).__init__(name=name, **kwargs)
        self.dense1 = Dense(60, activation="tanh")
        self.dense2 = Dense(num_classes, activation="sigmoid")
        self.softmax = Softmax()

    def call(self, inputs, training=None, mask=None):
        """
        Forward pass for the classifier head.
        :param inputs: Input data.
        :param training: N/A
        :param mask: N/A
        :return: Output from model.
        """
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.softmax(x)

    def get_config(self):
        """ Method inherited from superclass."""
        raise NotImplementedError("Not implemented")
