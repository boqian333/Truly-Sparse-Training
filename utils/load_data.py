import numpy as np
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor
from keras.datasets import cifar100, cifar10, mnist
from keras.utils import np_utils
from PIL import Image
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
from utils.tool import StandardScaler, to_categorical, onehot
from keras.datasets import fashion_mnist


class Error(Exception):
    pass


# Artificial dataset with two classes
def load_madelon_data():
    # Download the data
    x_train = np.loadtxt("data/Madelon/madelon_train.data")
    y_train = np.loadtxt('data/Madelon//madelon_train.labels')
    x_val = np.loadtxt('data/Madelon/madelon_valid.data')
    y_val = np.loadtxt('data/Madelon/madelon_valid.labels')
    x_test = np.loadtxt('data/Madelon/madelon_test.data')

    y_train = np.where(y_train == -1, 0, 1)
    y_val = np.where(y_val == -1, 0, 1)

    xTrainMean = np.mean(x_train, axis=0)
    xTtrainStd = np.std(x_train, axis=0)
    x_train = (x_train - xTrainMean) / xTtrainStd
    x_test = (x_test - xTrainMean) / xTtrainStd
    x_val = (x_val - xTrainMean) / xTtrainStd

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_val = x_val.astype('float32')
    y_train = np_utils.to_categorical(y_train, 2)
    y_val = np_utils.to_categorical(y_val, 2)

    return x_train, y_train, x_val, y_val


# The MNIST database of handwritten digits.
def load_mnist_data(n_training_samples, n_testing_samples):

    # read CIFAR10 data
    (x, y), (x_test, y_test) = mnist.load_data()

    y = np_utils.to_categorical(y, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x = x.astype('float32')
    x_test = x_test.astype('float32')

    index_train = np.arange(x.shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(x_test.shape[0])
    np.random.shuffle(index_test)

    x_train = x[index_train[0:n_training_samples], :]
    y_train = y[index_train[0:n_training_samples], :]

    x_test = x_test[index_test[0:n_testing_samples], :]
    y_test = y_test[index_test[0:n_testing_samples], :]

    # Normalize data
    x_train = x_train / 255.
    x_test = x_test / 255.
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    return x_train, y_train, x_test, y_test


# Fashion-MNIST is a dataset of Zalando's article images consisting of a training set of 60,000 examples and a test set of 10,000 examples.
# Each example is a 28x28 grayscale image, associated with a label from 10 classes.
def load_fashion_mnist_data_ori_not_flattened(n_training_samples, n_testing_samples):

    data = np.load("data/FASHIONMNIST/fashion_mnist.npz")

    index_train = np.arange(data["X_train"].shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(data["X_test"].shape[0])
    np.random.shuffle(index_test)

    x_train = data["X_train"][index_train[0:n_training_samples], :]
    y_train = data["Y_train"][index_train[0:n_training_samples], :]
    x_test = data["X_test"][index_test[0:n_testing_samples], :]
    y_test = data["Y_test"][index_test[0:n_testing_samples], :]

    # Normalize in 0..1
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Normalize data
    x_train_mean = np.mean(x_train, axis=0)
    x_train_std = np.std(x_train, axis=0)
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    return x_train, y_train, x_test, y_test

def load_fashion_mnist_data_ori(n_training_samples, n_testing_samples):

    data = np.load("data/FASHIONMNIST/fashion_mnist.npz")

    index_train = np.arange(data["X_train"].shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(data["X_test"].shape[0])
    np.random.shuffle(index_test)

    x_train = data["X_train"][index_train[0:n_training_samples], :]
    y_train = data["Y_train"][index_train[0:n_training_samples], :]
    x_test = data["X_test"][index_test[0:n_testing_samples], :]
    y_test = data["Y_test"][index_test[0:n_testing_samples], :]

    # Normalize in 0..1
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Normalize data
    x_train_mean = np.mean(x_train, axis=0)
    x_train_std = np.std(x_train, axis=0)
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    return x_train, y_train, x_test, y_test

def load_fashion_mnist_data_not_flattened(n_training_samples, n_testing_samples):

    (x, y), (x_test, y_test) = fashion_mnist.load_data()

    y = np_utils.to_categorical(y, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    # Normalize in 0..1
    x = x.astype('float32')
    x_test = x_test.astype('float32')

    x = x[:, :, :, np.newaxis]
    x_test = x_test[:, :, :, np.newaxis]

    index_train = np.arange(x.shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(x_test.shape[0])
    np.random.shuffle(index_test)

    x_train = x[index_train[0:n_training_samples], :]
    y_train = y[index_train[0:n_training_samples], :]

    x_test = x_test[index_test[0:n_testing_samples], :]
    y_test = y_test[index_test[0:n_testing_samples], :]

    # Normalize data
    x_train_mean = np.mean(x_train, axis=0)
    x_train_std = np.std(x_train, axis=0)
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    return x_train, y_train, x_test, y_test

def load_fashion_mnist_data(n_training_samples, n_testing_samples):

    (x, y), (x_test, y_test) = fashion_mnist.load_data()

    y = np_utils.to_categorical(y, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    # Normalize in 0..1
    x = x.astype('float32')
    x_test = x_test.astype('float32')

    x = x[:, :, :, np.newaxis]
    x_test = x_test[:, :, :, np.newaxis]

    index_train = np.arange(x.shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(x_test.shape[0])
    np.random.shuffle(index_test)

    x_train = x[index_train[0:n_training_samples], :]
    y_train = y[index_train[0:n_training_samples], :]

    x_test = x_test[index_test[0:n_testing_samples], :]
    y_test = y_test[index_test[0:n_testing_samples], :]

    # Normalize data
    x_train_mean = np.mean(x_train, axis=0)
    x_train_std = np.std(x_train, axis=0)
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    # x_train = x_train.reshape(-1, 28 * 28 * 1)
    # x_test = x_test.reshape(-1, 28 * 28 * 1)

    return x_train, y_train, x_test, y_test


# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
# There are 50000 training images and 10000 test images.
def load_cifar10_data(n_training_samples, n_testing_samples):

    # read CIFAR10 data
    (x, y), (x_test, y_test) = cifar10.load_data()

    y = np_utils.to_categorical(y, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x = x.astype('float32')
    x_test = x_test.astype('float32')

    index_train = np.arange(x.shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(x_test.shape[0])
    np.random.shuffle(index_test)

    x_train = x[index_train[0:n_training_samples], :]
    y_train = y[index_train[0:n_training_samples], :]

    x_test = x_test[index_test[0:n_testing_samples], :]
    y_test = y_test[index_test[0:n_testing_samples], :]

    # Normalize data
    x_train_mean = np.mean(x_train, axis=0)
    x_train_std = np.std(x_train, axis=0)
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    x_train = x_train.reshape(-1, 32 * 32 * 3)
    x_test = x_test.reshape(-1, 32 * 32 * 3)

    return x_train, y_train, x_test, y_test


# Not flattened version of CIFAR10
def load_cifar10_data_not_flattened(n_training_samples, n_testing_samples):

    # read CIFAR10 data
    (x, y), (x_test, y_test) = cifar10.load_data()

    y = np_utils.to_categorical(y, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x = x.astype('float32')
    x_test = x_test.astype('float32')

    index_train = np.arange(x.shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(x_test.shape[0])
    np.random.shuffle(index_test)

    x_train = x[index_train[0:n_training_samples], :]
    y_train = y[index_train[0:n_training_samples], :]

    x_test = x_test[index_test[0:n_testing_samples], :]
    y_test = y_test[index_test[0:n_testing_samples], :]

    # Normalize data
    x_train_mean = np.mean(x_train, axis=0) # 50000, 32, 32, 3
    x_train_std = np.std(x_train, axis=0)
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    return x_train, y_train, x_test, y_test


def load_cifar100_data(n_training_samples, n_testing_samples):

    # read CIFAR10 data
    (x, y), (x_test, y_test) = cifar100.load_data()

    y = np_utils.to_categorical(y, 100)
    y_test = np_utils.to_categorical(y_test, 100)
    x = x.astype('float32')
    x_test = x_test.astype('float32')

    index_train = np.arange(x.shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(x_test.shape[0])
    np.random.shuffle(index_test)

    x_train = x[index_train[0:n_training_samples], :]
    y_train = y[index_train[0:n_training_samples], :]

    x_test = x_test[index_test[0:n_testing_samples], :]
    y_test = y_test[index_test[0:n_testing_samples], :]

    # Normalize data
    x_train_mean = np.mean(x_train, axis=0)
    x_train_std = np.std(x_train, axis=0)
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    x_train = x_train.reshape(-1, 32 * 32 * 3)
    x_test = x_test.reshape(-1, 32 * 32 * 3)

    return x_train, y_train, x_test, y_test

# Not flattened version of CIFAR100
def load_cifar100_data_not_flattened(n_training_samples, n_testing_samples):

    # read CIFAR10 data
    (x, y), (x_test, y_test) = cifar100.load_data()

    y = np_utils.to_categorical(y, 100)
    y_test = np_utils.to_categorical(y_test, 100)
    x = x.astype('float32')
    x_test = x_test.astype('float32')

    index_train = np.arange(x.shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(x_test.shape[0])
    np.random.shuffle(index_test)

    x_train = x[index_train[0:n_training_samples], :]
    y_train = y[index_train[0:n_training_samples], :]

    x_test = x_test[index_test[0:n_testing_samples], :]
    y_test = y_test[index_test[0:n_testing_samples], :]

    # Normalize data
    x_train_mean = np.mean(x_train, axis=0)
    x_train_std = np.std(x_train, axis=0)
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    return x_train, y_train, x_test, y_test


def load_shape_data(n_samples, n_class, path2imgs, path2masks, valid_size, im_size=32, one_hot=False):

    # read shape data
    imgsmap = np.memmap(path2imgs, dtype=np.uint8, mode='r+', shape=(n_samples, im_size, im_size, 3))
    masksmap = np.memmap(path2masks, dtype=np.uint8, mode='r+', shape=(n_samples, im_size, im_size))
    imgarr = np.array(imgsmap)
    maskarr = np.array(masksmap)
    x_train = imgarr[:-valid_size]
    y_train_ori = maskarr[:-valid_size]

    x_test = imgarr[-valid_size:]
    y_test_ori = maskarr[-valid_size:]

    if one_hot:
        y_train = onehot(y_train_ori, n_class)
        y_test = onehot(y_test_ori, n_class)
    else:
        y_train = y_train_ori
        y_test = y_test_ori

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    index_train = np.arange(x_train.shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(x_test.shape[0])
    np.random.shuffle(index_test)

    x_train_ = x_train[index_train, :]
    y_train_ = y_train[index_train, :]

    x_test_ = x_test[index_test, :]
    y_test_ = y_test[index_test, :]

    # Normalize data
    x_train_mean = np.mean(x_train_, axis=0)
    x_train_std = np.std(x_train_, axis=0) + 1e-8
    x_train_ = (x_train_ - x_train_mean) / x_train_std
    x_test_ = (x_test_ - x_train_mean) / x_train_std

    # x_train_ = x_train_ / 255.
    # x_test_ = x_test_ / 255.

    x_train_ = x_train_.reshape(-1, im_size * im_size * 3)
    x_test_ = x_test_.reshape(-1, im_size * im_size * 3)

    y_train_ = y_train_.reshape(-1, im_size * im_size * (n_class))
    y_test_ = y_test_.reshape(-1, im_size * im_size * (n_class))

    return x_train_, y_train_, x_test_, y_test_

class Dataset_Custom():
    def __init__(self, root_path, size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path

    def _read_data(self, flag='train'):
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns);
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        # data_stamp = time_features(df_stamp, timeenc=timeenc, freq=freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]

    def _get_data(self, flag='train'):
        self._read_data(flag=flag)

        self.data_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        self.y_len = len(self.data_y) - self.seq_len - self.pred_len + 1

        self.train_data = []
        self.train_label = []
        for index in range(self.data_len):
            s_begin = index
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[s_end:(s_end+self.pred_len)]
            self.train_data.append(seq_x)
            self.train_label.append(seq_y)

        self.train_data = np.concatenate(self.train_data, axis=1)
        self.train_label = np.concatenate(self.train_label, axis=1)
        self.train_data = self.train_data.transpose((1, 0))
        self.train_label = self.train_label.transpose((1, 0))

        return self.train_data, self.train_label, self.scaler

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)




