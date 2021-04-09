class SSL:
    """
    Class for Semi Supervised Learner (SSL), consisting of an autoencoder and a classifier.
    """

    def __init__(self, autoencoder, classifier):
        """
        Initializes variables.
        :param autoencoder: The autoencoder part of the SSL.
        :param classifier: The classifier part of the SSL.
        """
        self.autoencoder = autoencoder
        self.classifier = classifier

    def fit_autoencoder(self, x, epochs, batch_size, x_val):
        """
        Performs training for the autoencoder.
        :param x: Input images.
        :param epochs: Number of epochs to train.
        :param batch_size: Batch size to use during training.
        :param x_val: Validation data.
        :return: History object obtained from training.
        """
        return self.autoencoder.fit(x,
                                    x,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    validation_data=(x_val, x_val))

    def fit_classifier(self, x, y, epochs, batch_size, validation_data):
        """
        Performs training for the classifier.
        :param x: Input data.
        :param y: Labels encoded as one hot vectors.
        :param epochs: Number of epochs to use during training.
        :param batch_size: Batch size ot use during training.
        :param validation_data: Tuple of (x_val, y_val)
        :return: History object obtained from training.
        """
        return self.classifier.fit(x,
                                   y,
                                   epochs=epochs,
                                   batch_size=batch_size,
                                   validation_data=validation_data)

    def get_encoder(self):
        """
        :return: The encoder of the autoencoder.
        """
        return self.autoencoder.encoder

    def evaluate_classifier(self, x, y):
        """
        Evaluates the classifier.
        :param x: Input data.
        :param y: Labels encoded as one hot vectors.
        :return: History object.
        """
        return self.classifier.evaluate(x, y)

    def forward_ae(self, x):
        """
        Forward pass for the autoencoder.
        :param x: x
        :return: Output from the autoencoder.
        """
        return self.autoencoder(x)
