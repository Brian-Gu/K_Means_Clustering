
import numpy as np
from imageio import imread
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def fill_image(cluster, predicted, width, height):
    new = np.zeros([width, height, 3])
    for i in range(width):
        for j in range(height):
            for k in range(3):
                center = cluster.cluster_centers_[predicted[i][j], :]
                new[i][j][k] = center[k]
    return np.array(new, dtype=np.uint8)


def main():
    img = np.array(imread('trees.png'), dtype=np.float64)
    w, h, c = img.shape
    mat = np.reshape(img, (w * h, c))
    for k in [3, 5, 10, 15]:
        kmn = KMeans(k).fit(mat)
        prd = kmn.predict(mat).reshape(img.shape[0:2])
        res = fill_image(kmn, prd, w, h)
        plt.imshow(res)
        plt.show()


if __name__ == '__main__':
    main()
