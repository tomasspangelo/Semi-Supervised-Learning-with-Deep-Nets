import tensorflow as tf


class Classifier(tf.keras.Model):
    def __init__(self, encoder, classifier_head, name="classifier", **kwargs):
        super(Classifier, self).__init__(name=name, **kwargs)
        self.encoder = encoder
        self.classifier_head = classifier_head

    def call(self, inputs, training=None, mask=None):
        return

    def get_config(self):
        raise NotImplementedError("Not implemented")


class ClassifierHead(tf.keras.Model):
    def __init__(self, name="classifierhead", **kwargs):
        super(Classifier, self).__init__(name=name, **kwargs)

    def call(self, inputs, training=None, mask=None):
        return

    def get_config(self):
        raise NotImplementedError("Not implemented")

