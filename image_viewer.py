import matplotlib.pyplot as plt
import numpy as np


class ImageViewer:

    @staticmethod
    def view(images, n_cols=2):
        fig = plt.figure()
        n_rows = int(np.ceil(len(images) / n_cols))
        i = 1
        for image in images:
            a = fig.add_subplot(n_rows, n_cols, i)
            a.axis("off")
            plt.imshow(image)
            i += 1
        fig.show()
