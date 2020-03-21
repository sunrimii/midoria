import numpy as np


class Mnist:
    def __init__(self, dim=2):
        self.dim = dim
        
        self.training_images = self.load_images('database/train-images.idx3-ubyte')
        self.training_labels = self.load_labels('database/train-labels.idx1-ubyte')
        self.testing_images = self.load_images('database/t10k-images.idx3-ubyte')
        self.testing_labels = self.load_labels('database/t10k-labels.idx1-ubyte')

    def load_images(self, path):
        with open(path, 'rb') as file:
            file.read(16)
            if self.dim == 2:
                images = np.fromfile(file, dtype=np.uint8).reshape(-1, 28*28) / 255
                images = np.rot90(images)
                images = np.flipud(images)
            elif self.dim == 4:
                images = np.fromfile(file, dtype=np.uint8).reshape(-1, 28, 28, 1) / 255
            return images

    def load_labels(self, path):
        with open(path, 'rb') as file:
            file.read(8)
            labels = np.fromfile(file, dtype=np.uint8)
            one_hot_labels = np.zeros((10, len(labels)))
            for i, l in enumerate(labels):
                one_hot_labels[l, i] = 1
            return one_hot_labels
