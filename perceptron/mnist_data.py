# ！/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Jie Xu'

import struct
import numpy as np
import os
import matplotlib.pyplot as plt


def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
        print(labels.shape)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        print(images.shape)
    return images, labels


image, label = load_mnist('D:/Project/DeepLearning/perceptron/')



fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True, figsize=(15, 7))
ax = ax.flatten()
for i in range(25):
    img = image[i].reshape(28, 28)
    print(img)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()