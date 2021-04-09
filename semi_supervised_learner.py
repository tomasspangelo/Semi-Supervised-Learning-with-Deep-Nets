class SSL:
    def __init__(self, autoencoder, classifier):
        self.autoencoder = autoencoder
        self.classifier = classifier

    def fit_autoencoder(self, x, epochs, batch_size, validation_data):
        return self.autoencoder.fit(x,
                                    x,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    validation_data=validation_data)

    def fit_classifier(self, x, y, epochs, batch_size, validation_data):
        return self.classifier.fit(x,
                                   y,
                                   epochs=epochs,
                                   batch_size=batch_size,
                                   validation_data=validation_data)

    def get_encoder(self):
        return self.autoencoder.encoder

    def evaluate_classifier(self, x, y):
        return self.classifier.evaluate(x, y)

    def forward_ae(self, x):
        return self.autoencoder(x)
