import matplotlib.pyplot as plt
import numpy as np


class ImageViewer:
    """Class used statically to view images."""

    @staticmethod
    def view(images, n_cols=2):
        """
        Plots images.
        :param images: Numpy array containing images.
        :param n_cols: Number of columns in plot.
        :return: None
        """
        fig = plt.figure()
        n_rows = int(np.ceil(len(images) / n_cols))
        i = 1
        for image in images:
            a = fig.add_subplot(n_rows, n_cols, i)
            a.axis("off")
            plt.imshow(image, cmap="gray")
            i += 1
        fig.show()
