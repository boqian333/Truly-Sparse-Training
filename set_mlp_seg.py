# Code associated with IJCAI 2023 tutorial " Sparse Training for Supervised, Unsupervised, Continual, and Deep Reinforcement Learning with Deep Neural Networks"

# If you use parts of this code please cite the following article:
# @article{Mocanu2018SET,
#   author  =    {Mocanu, Decebal Constantin and Mocanu, Elena and Stone, Peter and Nguyen, Phuong H. and Gibescu, Madeleine and Liotta, Antonio},
#   journal =    {Nature Communications},
#   title   =    {Scalable Training of Artificial Neural Networks with Adaptive Sparse Connectivity inspired by Network Science},
#   year    =    {2018},
#   doi     =    {10.1038/s41467-018-04316-3}
# }

# If you have space please consider citing also these articles

# @phdthesis{Mocanu2017PhDthesis,
#   title     =    "Network computations in artificial intelligence",
#   author    =    "D.C. Mocanu",
#   year      =    "2017",
#   isbn      =    "978-90-386-4305-2",
#   publisher =    "Eindhoven University of Technology",
# }

# @article{Liu2019onemillion,
#   author  =    {Liu, Shiwei and Mocanu, Decebal Constantin and Mocanu and Ramapuram Matavalam, Amarsagar Reddy and Pei, Yulong Pei and Pechenizkiy, Mykola},
#   journal =    {arXiv:1901.09181},
#   title   =    {Sparse evolutionary Deep Learning with over one million artificial neurons on commodity hardware},
#   year    =    {2019},
# }

# We thank to:
# Thomas Hagebols: for performing a thorough analyze on the performance of SciPy sparse matrix operations
# Ritchie Vink (https://www.ritchievink.com): for making available on Github a nice Python implementation of fully connected MLPs. This SET-MLP implementation was built on top of his MLP code:
#                                             https://github.com/ritchie46/vanilla-machine-learning/blob/master/vanilla_mlp.py

from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import dok_matrix
from scipy.sparse import csr_matrix
from nn_functions import *

from utils.load_data import load_shape_data
from utils.monitor import Monitor
from dataloader import Data
from keras.preprocessing.image import ImageDataGenerator
import datetime
import os
import time
import json
import sys
import numpy as np
from numba import njit, prange
import argparse
from utils.score import SegmentationMetric
import torch
from utils.tool import onehot
from visual_shape_image import plot_and_save_images

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
sys.stderr = stderr

@njit(parallel=True, fastmath=True, cache=True)
def backpropagation_updates_numpy(a, delta, rows, cols, out):
    for i in prange(out.shape[0]):
        s = 0
        for j in range(a.shape[0]):
            s += a[j, rows[i]] * delta[j, cols[i]]
        out[i] = s / a.shape[0]


@njit(fastmath=True, cache=True)
def find_first_pos(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


@njit(fastmath=True, cache=True)
def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx


@njit(fastmath=True, cache=True)
def compute_accuracy(activations, y_test):
    correct_classification = 0
    for j in range(y_test.shape[0]):
        if np.argmax(activations[j]) == np.argmax(y_test[j]):
            correct_classification += 1
    return correct_classification / y_test.shape[0]


@njit(fastmath=True, cache=True)
def dropout(x, rate):
    noise_shape = x.shape
    noise = np.random.uniform(0., 1., noise_shape)
    keep_prob = 1. - rate
    scale = np.float32(1 / keep_prob)
    keep_mask = noise >= rate
    return x * scale * keep_mask, keep_mask


def create_sparse_weights(epsilon, n_rows, n_cols, weight_init):
    # He uniform initialization
    if weight_init == 'he_uniform':
        limit = np.sqrt(6. / float(n_rows))

    # Xavier initialization
    if weight_init == 'xavier':
        limit = np.sqrt(6. / (float(n_rows) + float(n_cols)))

    mask_weights = np.random.rand(n_rows, n_cols)
    prob = 1 - (epsilon * (n_rows + n_cols)) / (n_rows * n_cols)  # normal to have 8x connections

    # generate an Erdos Renyi sparse weights mask
    weights = lil_matrix((n_rows, n_cols))
    n_params = np.count_nonzero(mask_weights[mask_weights >= prob])
    weights[mask_weights >= prob] = np.random.uniform(-limit, limit, n_params)
    print("Create sparse matrix with ", weights.getnnz(), " connections and ",
          (weights.getnnz() / (n_rows * n_cols)) * 100, "% density level")
    weights = weights.tocsr()
    return weights

def array_intersect(a, b):
    # this are for array intersection
    n_rows, n_cols = a.shape
    dtype = {'names': ['f{}'.format(i) for i in range(n_cols)], 'formats': n_cols * [a.dtype]}
    return np.in1d(a.view(dtype), b.view(dtype))  # boolean return


class SET_MLP_SEG:
    def __init__(self, dimensions, activations, epsilon=20, weight_init='he_uniform', metric=None):
        """
        :param dimensions: (tpl/ list) Dimensions of the neural net. (input, hidden layer, output)
        :param activations: (tpl/ list) Activations functions.
        Example of three hidden layer with
        - 3312 input features
        - 3000 hidden neurons
        - 3000 hidden neurons
        - 3000 hidden neurons
        - 5 output classes
        layers -->    [1,        2,     3,     4,     5]
        ----------------------------------------
        dimensions =  (3312,     3000,  3000,  3000,  5)
        activations = (          Relu,  Relu,  Relu,  Sigmoid)
        """
        self.n_layers = len(dimensions)
        self.loss = None
        self.dropout_rate = 0.  # dropout rate
        self.learning_rate = None
        self.momentum = None
        self.weight_decay = None
        self.epsilon = epsilon  # control the sparsity level as discussed in the paper
        self.zeta = None  # the fraction of the weights removed
        self.dimensions = dimensions
        self.weight_init = weight_init
        self.save_filename = ""
        self.input_layer_connections = []
        self.monitor = None
        self.importance_pruning = False

        self.training_time = 0
        self.testing_time = 0
        self.evolution_time = 0
        self.metric = metric

        # Weights and biases are initiated by index. For a one hidden layer net you will have a w[1] and w[2]
        self.w = {}
        self.b = {}
        self.pdw = {}
        self.pdd = {}
        self.gradients_for_growth = {}

        # Activations are also initiated by index. For the example we will have activations[2] and activations[3]
        self.activations = {}
        for i in range(len(dimensions) - 1):

            self.w[i + 1] = create_sparse_weights(self.epsilon, dimensions[i], dimensions[i + 1],
                                                      weight_init=self.weight_init)  # create sparse weight matrices
            self.b[i + 1] = np.zeros(dimensions[i + 1], dtype='float32')
            self.activations[i + 2] = activations[i]

    def _feed_forward(self, x, drop=False):
        """
        Execute a forward feed through the network.
        :param x: (array) Batch of input data vectors.
        :return: (tpl) Node outputs and activations per layer. The numbering of the output is equivalent to the layer numbers.
        """
        # w(x) + b
        z = {}

        # activations: f(z)
        a = {1: x}  # First layer has no activations as input. The input x is the input.
        masks = {}

        for i in range(1, self.n_layers):
            z[i + 1] = a[i] @ self.w[i] + self.b[i]
            a[i + 1] = self.activations[i + 1].activation(z[i + 1])
            if drop:
                if i < self.n_layers - 1:
                    # apply dropout
                    a[i + 1], keep_mask = dropout(a[i + 1], self.dropout_rate)
                    masks[i + 1] = keep_mask

        return z, a, masks

    def _back_prop(self, z, a, masks, y_true):
        """
        The input dicts keys represent the layers of the net.
        a = { 1: x,
              2: f(w1(x) + b1)
              3: f(w2(a2) + b2)
              4: f(w3(a3) + b3)
              5: f(w4(a4) + b4)
              }
        :param z: (dict) w(x) + b
        :param a: (dict) f(z)
        :param y_true: (array) One hot encoded truth vector.
        :return:
        """
        keep_prob = 1.
        if self.dropout_rate > 0:
            keep_prob = np.float32(1. - self.dropout_rate)

        # Determine partial derivative and delta for the output layer.
        # delta output layer
        delta = self.loss.delta(y_true, a[self.n_layers])
        dw = coo_matrix(self.w[self.n_layers - 1], dtype='float32')
        # compute backpropagation updates
        backpropagation_updates_numpy(a[self.n_layers - 1], delta, dw.row, dw.col, dw.data)

        update_params = {
            self.n_layers - 1: (dw.tocsr(),  np.mean(delta, axis=0))
        }

        # In case of three layer net will iterate over i = 2 and i = 1
        # Determine partial derivative and delta for the rest of the layers.
        # Each iteration requires the delta from the previous layer, propagating backwards.
        for i in reversed(range(2, self.n_layers)):
            # dropout for the backpropagation step
            if keep_prob != 1:
                delta = (delta @ self.w[i].transpose()) * self.activations[i].prime(z[i])
                delta = delta * masks[i]
                delta /= keep_prob
            else:
                delta = (delta @ self.w[i].transpose()) * self.activations[i].prime(z[i])

            dw = coo_matrix(self.w[i - 1], dtype='float32')

            # compute backpropagation updates
            backpropagation_updates_numpy(a[i - 1], delta, dw.row, dw.col, dw.data)

            update_params[i - 1] = (dw.tocsr(),  np.mean(delta, axis=0))
        for k, v in update_params.items():
            self._update_w_b(k, v[0], v[1])

    def _update_w_b(self, index, dw, delta):
        """
        Update weights and biases.
        :param index: (int) Number of the layer
        :param dw: (array) Partial derivatives
        :param delta: (array) Delta error.
        """

        # perform the update with momentum
        if index not in self.pdw:
            self.pdw[index] = - self.learning_rate * dw
            self.pdd[index] = - self.learning_rate * delta
        else:
            self.pdw[index] = self.momentum * self.pdw[index] - self.learning_rate * dw
            self.pdd[index] = self.momentum * self.pdd[index] - self.learning_rate * delta

        self.w[index] += self.pdw[index] - self.weight_decay * self.w[index]
        self.b[index] += self.pdd[index] - self.weight_decay * self.b[index]

    def fit(self, data, loss, epochs, batch_size, learning_rate=1e-3, momentum=0.9,
            weight_decay=0.0002, zeta=0.3, dropoutrate=0., testing=True, save_filename="", monitor=False, importance_pruning=False, algo="set"):
        """
        :param x: (array) Containing parameters
        :param y_true: (array) Containing one hot encoded labels.
        :return (array) A 2D array of metrics (epochs, 3).
        """

        self.monitor = Monitor(save_filename=save_filename) if monitor else None

        # Initiate the loss object with the final activation function
        self.loss = loss()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.zeta = zeta
        self.dropout_rate = dropoutrate
        self.save_filename = save_filename
        self.input_layer_connections.append(self.get_core_input_connections())
        np.savez_compressed(self.save_filename + "_input_connections.npz",
                            inputLayerConnections=self.input_layer_connections)

        self.importance_pruning = importance_pruning
        self.data = data
        self.algo = algo

        maximum_accuracy = 0
        metrics = np.zeros((epochs, 4))

        batches_per_epoch = self.data.get_train_data().shape[0] // batch_size

        for i in range(epochs):
            if self.monitor:
                self.monitor.start_monitor()

            self.data.shuffle()

            # training
            t1 = datetime.datetime.now()

            batches = 0
            for x_batch, y_batch in data.generate_data():
                z, a, masks = self._feed_forward(x_batch, True)
                batches += 1
                if batches >= batches_per_epoch:
                    self._back_prop(z, a, masks, y_batch)
                    break
                else:
                    self._back_prop(z, a, masks, y_batch)

            t2 = datetime.datetime.now()

            if self.monitor:
                self.monitor.stop_monitor()

            print("\nSET-MLP-SEG Epoch ", i)
            print("Training time: ", t2 - t1)
            self.training_time += (t2 - t1).seconds

            # test model performance on the test data at each epoch
            # this part is useful to understand model performance and can be commented for production settings
            if testing:

                t3 = datetime.datetime.now()
                pixAcc_test, mIoU_test, activations_test = self.predict(self.data.get_test_data(), self.data.get_test_labels())
                pixAcc_train, mIoU_train, activations_train = self.predict(self.data.get_train_data(), self.data.get_train_labels())

                t4 = datetime.datetime.now()
                maximum_accuracy = max(maximum_accuracy, mIoU_test)
                loss_test = self.loss.loss(self.data.get_test_labels(), activations_test)
                loss_train = self.loss.loss(self.data.get_train_labels(), activations_train)
                metrics[i, 0] = loss_train
                metrics[i, 1] = loss_test
                metrics[i, 2] = mIoU_train
                metrics[i, 3] = mIoU_test

                # print(metrics)

                print(f"Testing time: {t4 - t3}\n; Loss train: {loss_train}; PixACC train: {pixAcc_train}; mIoU train: {mIoU_train}; \n"
                      f" Loss test: {loss_test}; PixACC test: {pixAcc_test}; mIoU test: {mIoU_test}; \n"
                      f"Maximum mIoU val: {maximum_accuracy}")
                self.testing_time += (t4 - t3).seconds

            t5 = datetime.datetime.now()
            if i < epochs - 1:  # do not change connectivity pattern after the last epoch
                if self.algo == "set":
                    self.weights_evolution_II(i)  # this implementation has the same behaviour as the one above, but it is much faster.
            t6 = datetime.datetime.now()
            print("Weights evolution time ", t6 - t5)

            # save performance metrics values in a file
            if self.save_filename != "":
                np.savetxt(self.save_filename +".txt", metrics)

            if self.save_filename != "" and self.monitor:
                with open(self.save_filename + "_monitor.json", 'w') as file:
                    file.write(json.dumps(self.monitor.get_stats(), indent=4, sort_keys=True, default=str))

        return metrics

    def get_core_input_connections(self):
        values = np.sort(self.w[1].data)
        first_zero_pos = find_first_pos(values, 0)
        last_zero_pos = find_last_pos(values, 0)

        largest_negative = values[int((1 - self.zeta) * first_zero_pos)]
        smallest_positive = values[
            int(min(values.shape[0] - 1, last_zero_pos + self.zeta * (values.shape[0] - last_zero_pos)))]

        wlil = self.w[1].tolil()
        wdok = dok_matrix((self.dimensions[0], self.dimensions[1]), dtype="float32")

        # remove the weights closest to zero
        keep_connections = 0
        for ik, (row, data) in enumerate(zip(wlil.rows, wlil.data)):
            for jk, val in zip(row, data):
                if (val < largest_negative) or (val > smallest_positive):
                    wdok[ik, jk] = val
                    keep_connections += 1
        return wdok.tocsr().getnnz(axis=1)

    def weights_evolution_I(self):
        # this represents the core of the SET procedure. It removes the weights closest to zero in each layer and add new random weights
        for i in range(1, self.n_layers - 1):

            values = np.sort(self.w[i].data)
            first_zero_pos = find_first_pos(values, 0)
            last_zero_pos = find_last_pos(values, 0)

            largest_negative = values[int((1 - self.zeta) * first_zero_pos)]
            smallest_positive = values[
                int(min(values.shape[0] - 1, last_zero_pos + self.zeta * (values.shape[0] - last_zero_pos)))]

            wlil = self.w[i].tolil()
            pdwlil = self.pdw[i].tolil()
            wdok = dok_matrix((self.dimensions[i - 1], self.dimensions[i]), dtype="float32")
            pdwdok = dok_matrix((self.dimensions[i - 1], self.dimensions[i]), dtype="float32")

            # remove the weights closest to zero
            keep_connections = 0
            for ik, (row, data) in enumerate(zip(wlil.rows, wlil.data)):
                for jk, val in zip(row, data):
                    if (val < largest_negative) or (val > smallest_positive):
                        wdok[ik, jk] = val
                        pdwdok[ik, jk] = pdwlil[ik, jk]
                        keep_connections += 1
            limit = np.sqrt(6. / float(self.dimensions[i]))

            # add new random connections
            for kk in range(self.w[i].data.shape[0] - keep_connections):
                ik = np.random.randint(0, self.dimensions[i - 1])
                jk = np.random.randint(0, self.dimensions[i])
                while (wdok[ik, jk] != 0):
                    ik = np.random.randint(0, self.dimensions[i - 1])
                    jk = np.random.randint(0, self.dimensions[i])
                wdok[ik, jk] = np.random.uniform(-limit, limit)
                pdwdok[ik, jk] = 0

            self.pdw[i] = pdwdok.tocsr()
            self.w[i] = wdok.tocsr()

    def weights_evolution_II(self, epoch=0):
        # this represents the core of the SET procedure. It removes the weights closest to zero in each layer and add new random weights
        # improved running time using numpy routines - Amarsagar Reddy Ramapuram Matavalam (amar@iastate.edu)
        for i in range(1, self.n_layers - 1):
            # uncomment line below to stop evolution of dense weights more than 80% non-zeros
            # if self.w[i].count_nonzero() / (self.w[i].get_shape()[0]*self.w[i].get_shape()[1]) < 0.8:
                t_ev_1 = datetime.datetime.now()

                # Importance Pruning
                if self.importance_pruning and epoch % 10 == 0 and epoch > 200:
                    sum_incoming_weights = np.abs(self.w[i]).sum(axis=0)
                    t = np.percentile(sum_incoming_weights, 20)
                    sum_incoming_weights = np.where(sum_incoming_weights <= t, 0, sum_incoming_weights)
                    ids = np.argwhere(sum_incoming_weights == 0)

                    weights = self.w[i].tolil()
                    pdw = self.pdw[i].tolil()
                    weights[:, ids[:,1]]=0
                    pdw[:, ids[:,1]] = 0

                    self.w[i] = weights.tocsr()
                    self.pdw[i] = pdw.tocsr()

                # converting to COO form - Added by Amar
                wcoo = self.w[i].tocoo()
                vals_w = wcoo.data
                rows_w = wcoo.row
                cols_w = wcoo.col

                pdcoo = self.pdw[i].tocoo()
                vals_pd = pdcoo.data
                rows_pd = pdcoo.row
                cols_pd = pdcoo.col
                # print("Number of non zeros in W and PD matrix before evolution in layer",i,[np.size(valsW), np.size(valsPD)])
                values = np.sort(self.w[i].data)
                first_zero_pos = find_first_pos(values, 0)
                last_zero_pos = find_last_pos(values, 0)

                largest_negative = values[int((1-self.zeta) * first_zero_pos)]
                smallest_positive = values[int(min(values.shape[0] - 1, last_zero_pos + self.zeta * (values.shape[0] - last_zero_pos)))]

                #remove the weights (W) closest to zero and modify PD as well
                vals_w_new = vals_w[(vals_w > smallest_positive) | (vals_w < largest_negative)]
                rows_w_new = rows_w[(vals_w > smallest_positive) | (vals_w < largest_negative)]
                cols_w_new = cols_w[(vals_w > smallest_positive) | (vals_w < largest_negative)]

                new_w_row_col_index = np.stack((rows_w_new, cols_w_new), axis=-1)
                old_pd_row_col_index = np.stack((rows_pd, cols_pd), axis=-1)

                new_pd_row_col_index_flag = array_intersect(old_pd_row_col_index, new_w_row_col_index)  # careful about order

                vals_pd_new = vals_pd[new_pd_row_col_index_flag]
                rows_pd_new = rows_pd[new_pd_row_col_index_flag]
                cols_pd_new = cols_pd[new_pd_row_col_index_flag]

                self.pdw[i] = coo_matrix((vals_pd_new, (rows_pd_new, cols_pd_new)), (self.dimensions[i - 1], self.dimensions[i])).tocsr()

                if i == 1:
                    self.input_layer_connections.append(coo_matrix((vals_w_new, (rows_w_new, cols_w_new)),
                                                                   (self.dimensions[i - 1], self.dimensions[i])).getnnz(axis=1))
                    np.savez_compressed(self.save_filename + "_input_connections.npz",
                                        inputLayerConnections=self.input_layer_connections)

                # add new random connections
                keep_connections = np.size(rows_w_new)
                length_random = vals_w.shape[0] - keep_connections
                if self.weight_init == 'normal':
                    random_vals = np.random.randn(length_random) / 10  # to avoid multiple whiles, can we call 3*rand?
                else:
                    if self.weight_init == 'he_uniform':
                        limit = np.sqrt(6. / float(self.dimensions[i - 1]))
                    if self.weight_init == 'xavier':
                        limit = np.sqrt(6. / (float(self.dimensions[i - 1]) + float(self.dimensions[i])))
                    random_vals = np.random.uniform(-limit, limit, length_random)

                # adding  (wdok[ik,jk]!=0): condition
                while length_random > 0:
                    ik = np.random.randint(0, self.dimensions[i - 1], size=length_random, dtype='int32')
                    jk = np.random.randint(0, self.dimensions[i], size=length_random, dtype='int32')

                    random_w_row_col_index = np.stack((ik, jk), axis=-1)
                    random_w_row_col_index = np.unique(random_w_row_col_index, axis=0)  # removing duplicates in new rows&cols
                    oldW_row_col_index = np.stack((rows_w_new, cols_w_new), axis=-1)

                    unique_flag = ~array_intersect(random_w_row_col_index, oldW_row_col_index)  # careful about order & tilda

                    ik_new = random_w_row_col_index[unique_flag][:,0]
                    jk_new = random_w_row_col_index[unique_flag][:,1]
                    # be careful - row size and col size needs to be verified
                    rows_w_new = np.append(rows_w_new, ik_new)
                    cols_w_new = np.append(cols_w_new, jk_new)

                    length_random = vals_w.shape[0]-np.size(rows_w_new) # this will constantly reduce lengthRandom

                # adding all the values along with corresponding row and column indices - Added by Amar
                vals_w_new = np.append(vals_w_new, random_vals) # be careful - we can add to an existing link ?
                # vals_pd_new = np.append(vals_pd_new, zero_vals) # be careful - adding explicit zeros - any reason??
                if vals_w_new.shape[0] != rows_w_new.shape[0]:
                    print("not good")
                self.w[i] = coo_matrix((vals_w_new, (rows_w_new, cols_w_new)), (self.dimensions[i-1], self.dimensions[i])).tocsr()

                # print("Number of non zeros in W and PD matrix after evolution in layer",i,[(self.w[i].data.shape[0]), (self.pdw[i].data.shape[0])])

                t_ev_2 = datetime.datetime.now()
                print("Weights evolution time for layer", i, "is", t_ev_2 - t_ev_1)
                self.evolution_time += (t_ev_2 - t_ev_1).seconds


    def predict(self, x_test, y_test, batch_size=100, save=False):
        """
        :param x_test: (array) Test input
        :param y_test: (array) Correct test output
        :param batch_size:
        :return: (flt) Classification accuracy
        :return: (array) A 2D array of shape (n_cases, n_classes).
        """
        self.metric.reset()
        pixAcc = 0.0
        mIoU = 0.0
        activations = np.zeros((y_test.shape[0], y_test.shape[1]))
        for j in range(x_test.shape[0] // batch_size):
            k = j * batch_size
            l = (j + 1) * batch_size
            _, a_test, _ = self._feed_forward(x_test[k:l], drop=False)
            activations[k:l] = a_test[self.n_layers]

            outputs = activations[k:l].reshape(batch_size, 32, 32)
            target = y_test[k:l].reshape(batch_size, 32, 32)

            outputs_out = np.where(outputs > 0.5, 1.0, 0.)
            target_out = np.where(target>0.5, 1.0, 0.)
            outputs_onehot = onehot(outputs_out, 1)

            ## plot output
            # if save:
            #     for index in range(20):
            #         original_image = x_test[index].reshape(32, 32, 3)
            #         target_image = target_out[index]
            #         outputs_image = outputs_out[index]
            #         plot_and_save_images(original_image, target_image, outputs_image, index=index)
            #     break

            metric.update(torch.tensor(outputs_onehot), torch.tensor(target_out))
        pixAcc, mIoU = metric.get()

        # print("Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(pixAcc, mIoU))
        return pixAcc, mIoU, activations

def str2bool(str):
    return True if str.lower() == 'true' else False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sequential SET-MLP config')

    parser.add_argument('--monitor', help='Monitor cpu and gpu utilization', default=True, action='store_true')

    # Configuration of training process
    parser.add_argument('--loss', help='loss function', default='cross_entropy')

    # logging configuration
    parser.add_argument('--log-file', default=None, dest='log_file',
                        help='log file to write, in additon to output stream')
    parser.add_argument('--log-level', default='info', dest='log_level', help='log level (debug, info, warn, error)')

    # Model configuration
    parser.add_argument('--batch-size', type=int, default=100, help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=400, help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate (default: 0.01)')
    parser.add_argument('--lr-rate-decay', type=float, default=0.0, help='learning rate decay (default: 0)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--dropout-rate', type=float, default=0.3, help='Dropout rate (default: 0.3)')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay (l2 regularization)')
    parser.add_argument('--epsilon', type=int, default=20, help='Sparsity level (default: 20)')
    parser.add_argument('--zeta', type=float, default=0.3,
                        help='It gives the percentage of unimportant connections which are removed and replaced with '
                             'random ones after every epoch(in [0..1])')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--n-training-samples', type=int, default=60000, help='Number of training samples')
    parser.add_argument('--n-testing-samples', type=int, default=10000, help='Number of testing samples')
    parser.add_argument('--dataset', default='seg_shape32', help='Specify dataset. One of "cifar10", "fashionmnist",' '"madelon",  or "mnist"')

    parser.add_argument("--augmentation", type=str2bool, default=False, help='Data augmentation')
    parser.add_argument("--prune", type=str2bool, default=False, help='Perform Importance Pruning')
    parser.add_argument("--allrelu", type=str2bool, default=False, help='use allrelu or relu')

    parser.add_argument('--algo', type=str, default='set', help='sparse training algorithm"')
    args = parser.parse_args()
    print(args)

    sum_training_time = 0
    runs = 1
    datagen = None

    metric = SegmentationMetric(nclass=1)
    path2imgs = './data/data_32/shape_imgs_shape_1_500.dat'
    path2masks = './data/data_32/shape_masks_shape_1_500.dat'
    valid_size = 100

    for i in range(runs):

        if args.dataset == 'seg_shape32':
            # Model architecture cifar10
            dimensions = (3*32*32, 4000, 1000, 4000, 32*32)
            weight_init = 'he_uniform'
            loss = 'cross_entropy'

            if args.allrelu:
                activations = (AlternatedLeftReLU(-0.75), AlternatedLeftReLU(0.75), AlternatedLeftReLU(-0.75), Softmax)
            else:
                activations = (Relu, Relu, Relu, Sigmoid)


            X_train, Y_train, X_test, Y_test = load_shape_data(500, 1, path2imgs, path2masks, valid_size)
        else:
            raise NotImplementedError("The given dataset is not available")


        # set model parameters
        # no_hidden_neurons_layer = 1000
        epsilon = args.epsilon  # set the sparsity level
        zeta = args.zeta  # in [0..1]. It gives the percentage of unimportant connections which are removed and replaced with random ones after every epoch
        no_training_epochs = args.epochs
        batch_size = args.batch_size
        dropout_rate = args.dropout_rate
        learning_rate = args.lr
        momentum = args.momentum
        weight_decay = args.weight_decay
        importance_pruning = args.prune

        ## data new
        data = Data(batch_size=batch_size, x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test,
                    augmentation=args.augmentation)

        np.random.seed(i)

        # create SET-MLP-SEG (MLP with adaptive sparse connectivity trained with Sparse Evolutionary Training)
        set_mlp_seg = SET_MLP_SEG(dimensions=dimensions, activations=activations, epsilon=epsilon, metric=metric)

        start_time = time.time()
        # train SET-MLP-SEG
        set_mlp_seg.fit(data, loss=MSE, epochs=no_training_epochs, batch_size=batch_size, learning_rate=learning_rate,
                    momentum=momentum, weight_decay=weight_decay, zeta=zeta, dropoutrate=dropout_rate, testing=True,
                    save_filename="Results/set_mlp_seg_" + str(args.n_training_samples) + "_training_samples_e" + str(epsilon) + "_rand" + str(i), monitor=True, importance_pruning=importance_pruning,algo=args.algo)

        step_time = time.time() - start_time
        print("\nTotal execution time: ", step_time)
        print("\nTotal training time: ", set_mlp_seg.training_time)
        print("\nTotal testing time: ", set_mlp_seg.testing_time)
        print("\nTotal evolution time: ", set_mlp_seg.evolution_time)
        sum_training_time += step_time

        # test SET-MLP-SEG
        accuracy, _, _ = set_mlp_seg.predict(data.get_test_data(), data.get_test_labels(), batch_size=100)

        print("\nAccuracy of the last epoch on the testing data: ", accuracy)
    print(f"Average training time over {runs} runs is {sum_training_time/runs} seconds")
