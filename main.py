import sys
from configparser import ConfigParser
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

import numpy as np

from classifier import Classifier
from autoencoder import Autoencoder, Encoder
from semi_supervised_learner import SSL

from image_viewer import ImageViewer
from utils import convert_to_grayscale, tsne, load_kmnist, load_emnist, loss_from_string, optimizer_from_string
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def init_data(data_config):
    """
    Initializes the data.
    :param data_config: Data config.
    :return: The processed data, ready to be used for training.
    """
    dataset = data_config["name"]
    d1 = float(data_config["d1"])
    d1_train = float(data_config["d1_train"])
    d2_train = float(data_config["d2_train"])
    d2_val = float(data_config["d2_val"])
    size = int(data_config['size'])

    (x_train, y_train), (x_test, y_test) = (None, None), (None, None)
    num_classes = 0
    rgb = False
    if dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data(path="mnist.npz")
        num_classes = 10
    elif dataset == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        num_classes = 10
    elif dataset == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        rgb = True
        num_classes = 10
    elif dataset == "kmnist":
        (x_train, y_train), (x_test, y_test) = load_kmnist()
        num_classes = 10
    elif dataset == "emnist":
        (x_train, y_train), (x_test, y_test) = load_emnist()
        num_classes = 26

    input_shape = x_train.shape[1:]
    x = np.concatenate((x_train, x_test), axis=0)
    x = x[:size]
    if rgb:
        x = convert_to_grayscale(x)
    y = np.concatenate((y_train, y_test), axis=0)
    y = y[:size]
    x = x.reshape(x.shape + (1,))

    x = tf.cast(x, tf.float32) / 255.0

    y = to_categorical(y)

    n_d1 = int(np.ceil(len(x) * d1))
    x_d1 = x[:n_d1]
    y_d1 = y[:n_d1]
    x_d2 = x[n_d1:]
    y_d2 = y[n_d1:]

    n_d1_train = int(len(x_d1) * d1_train)
    x_d1_train = x_d1[:n_d1_train]
    x_d1_val = x_d1[n_d1_train:]

    y_d1_train = y_d1[:n_d1_train]
    y_d1_val = y_d1[n_d1_train:]

    n_d2_train = int(len(x_d2) * d2_train)
    n_d2_val = int(np.ceil(len(x_d2) * d2_val))

    x_d2_train = x_d2[:n_d2_train]
    x_d2_val = x_d2[n_d2_train:n_d2_train + n_d2_val]
    x_d2_test = x_d2[n_d2_train + n_d2_val:]

    y_d2_train = y_d2[:n_d2_train]
    y_d2_val = y_d2[n_d2_train:n_d2_train + n_d2_val]
    y_d2_test = y_d2[n_d2_train + n_d2_val:]

    d1 = (x_d1_train, x_d1_val, y_d1_train, y_d1_val)
    d2 = (x_d2_train, x_d2_val, x_d2_test, y_d2_train, y_d2_val, y_d2_test)
    return d1, d2, num_classes, input_shape


def init_autoencoder(ae_config, input_shape):
    """
    Initializes the autoencoder.
    :param ae_config: Autoencoder Config.
    :param input_shape: Shape of the images.
    :return: Autoencoder object.
    """
    latent_size = int(ae_config["latent_size"])

    ae = Autoencoder(image_shape=input_shape, latent_size=latent_size)

    loss_name = ae_config["loss"]
    loss = loss_from_string(loss_name)

    learning_rate = float(ae_config["learning_rate"])

    optimizer_name = ae_config["optimizer"]
    optimizer = optimizer_from_string(optimizer_name, learning_rate)

    ae.compile(optimizer=optimizer, loss=loss)

    return ae


def init_classifier(classifier_config, encoder, num_classes, freeze):
    """
    Initializes classifier.
    :param classifier_config: Classifier Config.
    :param encoder: Encoder to be used in classifier.
    :param num_classes: The number of different classes for the data.
    :param freeze: True if encoder should not learn, False otherwise.
    :return: Classifier object.
    """
    classifier = Classifier(num_classes=num_classes, encoder=encoder)

    optimizer_name = classifier_config["optimizer"]
    learning_rate = float(classifier_config["learning_rate"])
    optimizer = optimizer_from_string(optimizer_name, learning_rate)

    loss_name = classifier_config["loss"]
    loss = loss_from_string(loss_name)

    if freeze:
        classifier.encoder.freeze_model()
    classifier.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    return classifier


def main():
    """Main method for the system."""

    # Check if config file argument is provided.
    if len(sys.argv) < 2:
        print("Please indicate config file from /Config and try again.")
        return

    config = ConfigParser()
    config.read("./config/" + sys.argv[1])

    vis_config = config["visualization"]
    plot_tsne = vis_config.getboolean("tSNE")
    num = int(vis_config["num"])

    # Initialize the data
    data_config = config["dataset"]
    d1, d2, num_classes, input_shape = init_data(data_config)
    x_d1_train, x_d1_val, y_d1_train, y_d1_val = d1
    x_d2_train, x_d2_val, x_d2_test, y_d2_train, y_d2_val, y_d2_test = d2

    # Initialize the autoencoder
    ae_config = config["autoencoder"]
    autoencoder = init_autoencoder(ae_config, input_shape)  # Should not be used

    # Initialize semi-supervised learner
    classifier_config = config["classifier"]
    freeze = classifier_config.getboolean("freeze")

    ssl = SSL(autoencoder=autoencoder,
              classifier=init_classifier(classifier_config, autoencoder.encoder, num_classes, freeze))

    # Plot tSNE before any training
    if plot_tsne:
        tsne_fig1 = tsne(x_d1_train[:num], y_d1_train[:num], 1, ssl.get_encoder())

    # Train autoencoder-part of semi-supervised learner
    epochs = int(ae_config["epochs"])
    batch_size = int(ae_config["batch_size"])
    print("Training autoencoder:")
    ae_hist = ssl.fit_autoencoder(x_d1_train,
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  x_val=x_d1_val)

    # Plot tSNE after autoencoder training
    if plot_tsne:
        tsne_fig2 = tsne(x_d1_train[:num], y_d1_train[:num], 2, ssl.get_encoder())

    # Plot reconstructions
    reconstructions = int(vis_config["reconstructions"])
    ImageViewer.view(x_d1_train[:reconstructions], n_cols=4)
    ImageViewer.view(ssl.forward_ae(x_d1_train[:reconstructions]), n_cols=4)

    # Plot autoencoder loss
    fig = plt.figure()
    plt.plot(ae_hist.history["loss"], 'b', label="loss")
    plt.plot(ae_hist.history["val_loss"], 'g', label="val_loss")
    plt.xlabel("Epoch")
    plt.title("Autoencoder Learning")
    plt.legend(loc="upper right")
    fig.show()

    # Train classifier part of semi-supervised learner
    epochs = int(classifier_config["epochs"])
    batch_size = int(classifier_config["batch_size"])

    print("Training C1:")
    hist1 = ssl.fit_classifier(x_d2_train,
                               y_d2_train,
                               epochs=epochs,
                               batch_size=batch_size,
                               validation_data=(x_d2_val, y_d2_val))
    print("Evaluating C1 on D2 test set:")
    ssl.evaluate_classifier(x_d2_test, y_d2_test)

    # Initialize purely supervised classifier
    latent_size = ae_config["latent_size"]
    classifier2 = init_classifier(classifier_config, Encoder(latent_size), num_classes, freeze)

    # Train purely supervised classifier
    print("Training C2:")
    hist2 = classifier2.fit(x_d2_train,
                            y_d2_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(x_d2_val, y_d2_val))
    print("Evaluating C2 on D2 test set:")
    classifier2.evaluate(x_d2_test, y_d2_test)

    print("Evaluating C1 on D1 training set:")
    ssl.evaluate_classifier(x_d1_train, y_d1_train)

    print("Evaluating C2 on D1 training set:")
    classifier2.evaluate(x_d1_train, y_d1_train)

    # Plot classifier (both semi and supervised) accuracy from training
    fig = plt.figure()
    plt.plot(hist1.history["accuracy"], 'b', label="Semi-accuracy")
    plt.plot(hist1.history["val_accuracy"], 'orange', label="Semi-val_accuracy")
    plt.plot(hist2.history["accuracy"], 'g', label="Sup-accuracy")
    plt.plot(hist2.history["val_accuracy"], 'r', label="Sup-val_accuracy")
    plt.xlabel("Epoch")
    plt.title("Comparative Classifier Learning")
    plt.legend(loc="lower right")
    fig.show()

    # Plot tSNE after semi-supervised learner has trained both autoencoder
    # and classifier.
    if plot_tsne:
        tsne_fig3 = tsne(x_d1_train[:num], y_d1_train[:num], 3, ssl.get_encoder())
        tsne_fig1.show()
        tsne_fig2.show()
        tsne_fig3.show()


if __name__ == "__main__":
    main()
