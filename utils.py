import numpy as np
from PIL import Image
from matplotlib import colors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from image_viewer import ImageViewer


def convert_to_grayscale(arr):
    out = []
    for img_arr in arr:
        img = img_arr.astype(np.uint8)
        img = Image.fromarray(img).convert('L')
        img = np.array(img)
        out.append(img)
    return np.array(out)


# TODO: Add more colors
def tsne(x, y, num, encoder):
    latents = encoder(x)
    y = np.array([np.where(one_hot == 1)[0] for one_hot in y]).reshape(y.shape[0])
    color_list = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                  "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
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
    with np.load("./datasets/kmnist-train-imgs.npz") as data:
        x = data['arr_0']
    with np.load("./datasets/kmnist-train-labels.npz") as data:
        y = data['arr_0']
    return x, y


if __name__ == "__main__":
    pass


