import numpy as np
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy import io as spio


def convert_to_grayscale(arr):
    """
    Converts a RBG image to greyscale.
    :param arr: Array containing images.
    :return: Numpy array containing images in greyscale.
    """
    out = []
    for img_arr in arr:
        img = img_arr.astype(np.uint8)
        img = Image.fromarray(img).convert('L')
        img = np.array(img)
        out.append(img)
    return np.array(out)


def tsne(x, y, num, encoder):
    """
    Produces figure containing tSNE plot.
    :param x: Data
    :param y: Labels encoded as one hot vectors.
    :param num: 1, 2 or 3 depending on when in training.
    :param encoder: The encoder to produce latent vector.
    :return: Figure containing tSNE plot.
    """
    latents = encoder(x)
    y = np.array([np.where(one_hot == 1)[0] for one_hot in y]).reshape(y.shape[0])
    color_list = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                  "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
                  "dimgray", "maroon", "gold", "lime", "lightsteelblue", "plum",
                  "lightcyan", "tan", "yellow", "dodgerblue", "crimson", "hotpink",
                  "mediumseagreen", "papayawhip", "mistyrose", "indigo"]
    latents_embedded = TSNE(n_components=2, random_state=0).fit_transform(latents)
    fig = plt.figure()
    titles = ["tSNE prior to training", "tSNE after autoencoder training",
              "tSNE after autoencoder training and classifier training"]
    for i in range(len(latents_embedded)):
        latent = latents_embedded[i]
        plt.plot(latent[0], latent[1], color=color_list[y[i]], marker="o")
        plt.title(titles[num - 1])
    return fig


def load_kmnist():
    """
    :return: KMNIST dataset.
    """
    with np.load("./datasets/kmnist-train-imgs.npz") as data:
        x_train = data['arr_0']
    with np.load("./datasets/kmnist-test-imgs.npz") as data:
        x_test = data['arr_0']
    with np.load("./datasets/kmnist-train-labels.npz") as data:
        y_train = data['arr_0']
    with np.load("./datasets/kmnist-test-labels.npz") as data:
        y_test = data['arr_0']

    return (x_train, y_train), (x_test, y_test)


def load_emnist():
    """
    :return: EMNIST Alphabetic dataset.
    """
    emnist = spio.loadmat("./datasets/emnist-letters.mat")

    x_train = emnist["dataset"][0][0][0][0][0][0]
    x_train = x_train.astype(np.float32)

    y_train = emnist["dataset"][0][0][0][0][0][1]
    y_train = y_train.reshape(y_train.shape[0]) - 1

    x_test = emnist["dataset"][0][0][1][0][0][0]
    x_test = x_test.astype(np.float32)

    y_test = emnist["dataset"][0][0][1][0][0][1]
    y_test = y_test.reshape(y_test.shape[0]) - 1

    x_train = x_train.reshape(x_train.shape[0], 28, 28, order="A")
    x_test = x_test.reshape(x_test.shape[0], 28, 28, order="A")

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    pass
