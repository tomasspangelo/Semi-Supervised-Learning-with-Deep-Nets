import sys
from configparser import ConfigParser
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

import numpy as np

from classifier import Classifier
from autoencoder import Autoencoder, Encoder

from image_viewer import ImageViewer
from utils import convert_to_grayscale, tsne


def example():
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path="mnist.npz")
    # (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.reshape(x_train.shape + (1,))[:100]

    # x_train = tf.image.decode_jpeg(x_train)
    x_train = tf.cast(x_train, tf.float32) / 255.0
    ImageViewer.view(x_train[0:10])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.MeanSquaredError()
    ae = Autoencoder(image_shape=(28, 28), latent_size=40)
    ae.compile(optimizer=optimizer, loss=loss)
    ae.fit(x_train, x_train, epochs=250, batch_size=5)
    ImageViewer.view(ae(x_train[0:10]))

    classifier = Classifier(num_classes=10, encoder=ae.encoder)
    y_train = to_categorical(y_train)[:100]

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.CategoricalCrossentropy()
    # ae.encoder.freeze_model()
    classifier.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    classifier.fit(x_train, y_train, epochs=10, batch_size=2)


def loss_from_string(name):
    loss = None
    if name == "mse":
        loss = tf.keras.losses.MeanSquaredError()
    elif name == "cc":
        loss = tf.keras.losses.CategoricalCrossentropy()
    elif name == "bc":
        loss = tf.keras.losses.BinaryCrossentropy()
    elif name == "scc":
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
    elif name == "kld":
        loss = tf.keras.losses.KLDivergence()
    elif name == "mae":
        loss = tf.keras.losses.MeanAbsoluteError()
    return loss


def optimizer_from_string(name, learning_rate):
    if name == "adagrad":
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif name == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif name == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    return optimizer


# TODO: Ensure same number from classes
# TODO: Find 2 additional datasets
def init_data(data_config):
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
    if len(sys.argv) < 2:
        print("Please indicate config file from /Config and try again.")
        return

    config = ConfigParser()
    config.read("./config/" + sys.argv[1])

    vis_config = config["visualization"]
    plot_tsne = vis_config.getboolean("tSNE")

    data_config = config["dataset"]
    d1, d2, num_classes, input_shape = init_data(data_config)
    x_d1_train, x_d1_val, y_d1_train, y_d1_val = d1
    x_d2_train, x_d2_val, x_d2_test, y_d2_train, y_d2_val, y_d2_test = d2

    ae_config = config["autoencoder"]
    autoencoder = init_autoencoder(ae_config, input_shape)

    if plot_tsne:
        tsne_fig1 = tsne(x_d1_train[:200], y_d1_train[:200], 1, autoencoder.encoder)

    epochs = int(ae_config["epochs"])
    batch_size = int(ae_config["batch_size"])
    ae_hist = autoencoder.fit(x_d1_train,
                              x_d1_train,
                              epochs=epochs,
                              batch_size=batch_size,
                              validation_data=(x_d1_val, x_d1_val))

    if plot_tsne:
        tsne_fig2 = tsne(x_d1_train[:200], y_d1_train[:200], 2, autoencoder.encoder)

    reconstructions = int(vis_config["reconstructions"])
    ImageViewer.view(x_d1_train[:reconstructions], n_cols=4)
    ImageViewer.view(autoencoder(x_d1_train[:reconstructions]), n_cols=4)

    fig = plt.figure()
    plt.plot(ae_hist.history["loss"], 'b', label="loss")
    plt.plot(ae_hist.history["val_loss"], 'g', label="val_loss")
    plt.xlabel("Epoch")
    plt.title("Autoencoder Learning")
    plt.legend(loc="upper right")
    fig.show()

    classifier_config = config["classifier"]
    freeze = classifier_config.getboolean("freeze")
    classifier1 = init_classifier(classifier_config, autoencoder.encoder, num_classes, freeze)

    epochs = int(classifier_config["epochs"])
    batch_size = int(classifier_config["batch_size"])

    hist1 = classifier1.fit(x_d2_train,
                            y_d2_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(x_d2_val, y_d2_val))

    test_loss1, test_acc1 = classifier1.evaluate(x_d2_test, y_d2_test)

    classifier_config = config["classifier"]
    latent_size = ae_config["latent_size"]
    classifier2 = init_classifier(classifier_config, Encoder(latent_size), num_classes, freeze=False)
    hist2 = classifier2.fit(x_d2_train,
                            y_d2_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(x_d2_val, y_d2_val))
    test_loss2, test_acc2 = classifier2.evaluate(x_d2_test, y_d2_test)

    # TODO: Test classifier1 and classifier2 on x_d1 (labels: y_d1)

    fig = plt.figure()
    plt.plot(hist1.history["accuracy"], 'b', label="Semi-accuracy")
    plt.plot(hist1.history["val_accuracy"], 'orange', label="Semi-val_accuracy")
    plt.plot(hist2.history["accuracy"], 'g', label="Sup-accuracy")
    plt.plot(hist2.history["val_accuracy"], 'r', label="Sup-val_accuracy")
    plt.xlabel("Epoch")
    plt.title("Comparative Classifier Learning")
    plt.legend(loc="lower right")
    fig.show()

    if plot_tsne:
        tsne_fig3 = tsne(x_d1_train[:200], y_d1_train[:200], 3, autoencoder.encoder)
        tsne_fig1.show()
        tsne_fig2.show()
        tsne_fig3.show()


if __name__ == "__main__":
    main()
