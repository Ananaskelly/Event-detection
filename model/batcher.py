import numpy as np


from keras.utils import Sequence, to_categorical


class Batcher(Sequence):
    def __init__(self, x, y=None, batch_size=64):
        self.batch_size = batch_size
        self.x = x
        self.y = y
        self.indexes = np.arange(len(x))

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        indexes_tmp = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        if self.y is not None:
            res_x = self.x[indexes_tmp, :, :, np.newaxis]
            res_y = to_categorical(self.y, 41)[indexes_tmp]
            return res_x, res_y
        else:
            res_x = self.x[indexes_tmp, :, :, np.newaxis]
            return res_x
