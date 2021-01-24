import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC
import numpy as np
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_regression
import gc


def binary_svm(x_train, y_train, c=10, gamma=4):
    clf = SVC(kernel='rbf', C=c, gamma=gamma)
    clf.fit(x_train, y_train)
    return clf


def mutual_info_weight(x, label):
    mi = mutual_info_regression(x, label)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(1, 2))
    mi_scale = min_max_scaler.fit_transform(mi.reshape(-1, 1))
    mi_scale = mi_scale.ravel()
    mi = np.diag(mi_scale)
    return mi


class SupervisedSparseAutoencoder(object):

    def __init__(self, n_input, n_hidden, kl_reg,
                 regular_reg, sample_size, rho,
                 reconstruct_reg, activation=tf.nn.sigmoid):
        self.regular_reg = regular_reg
        self.sample_size = sample_size
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.rec_reg = reconstruct_reg
        self.kl_reg = kl_reg
        self.rho = rho
        self.activation = activation
        self.X = tf.placeholder("float", [None, self.n_input], name='X')
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.mutual_info = tf.placeholder("float", [self.n_input, self.n_input], name='mutual_info')

        self.weights = {
            'encoder_h1': tf.Variable(self.initializer([self.n_input, self.n_hidden]), name='encoder_h1'),
            'decoder_h1': tf.Variable(self.initializer([self.n_hidden, self.n_input]), name='decoder_h1'),
        }

        self.biases = {
            'encoder_b1': tf.Variable(self.initializer([self.n_hidden]), name='encoder_b1'),
            'decoder_b1': tf.Variable(self.initializer([self.n_input]), name='decoder_b1'),
        }
        self.encoder_op = self.encoder(self.X)
        self.decoder_op = self.decode(self.encoder_op)
        self.loss = self.cost(self.X, self.encoder_op, self.decoder_op)
        self.global_step = tf.Variable(0, trainable=False)
        self.lr = tf.train.exponential_decay(.001, self.global_step, 2000, 0.95, staircase=True)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.loss,
                                                                                   global_step=self.global_step)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def encoder(self, x):
        layer_1 = self.activation(tf.add(tf.matmul(x, self.weights['encoder_h1']),
                                         self.biases['encoder_b1']))
        return layer_1

    def decode(self, x):
        layer_1 = self.activation(tf.add(tf.matmul(x, self.weights['decoder_h1']),
                                         self.biases['decoder_b1']))
        return layer_1

    def regularization(self, ):
        return tf.nn.l2_loss(self.weights['encoder_h1']) + tf.nn.l2_loss(self.weights['decoder_h1'])

    def kl_divergence(self, rho, rho_hat):
        return rho * tf.log(rho) - rho * tf.log(rho_hat) + (1 - rho) * tf.log(1 - rho) - (1 - rho) * tf.log(1 - rho_hat)

    def cost(self, X, X_hat, X_):
        rho_hat = tf.reduce_mean(X_hat)
        kl_d = self.kl_divergence(self.rho, rho_hat)
        diff = X - X_
        diff = tf.matmul(diff ** 2, self.mutual_info)
        diff = tf.reduce_mean(tf.reduce_sum(diff, axis=1))
        loss = self.rec_reg * diff \
               + self.kl_reg * kl_d \
               + self.regular_reg * self.regularization()
        return loss

    def partial_fit(self, X, mutual_info):
        cost, _ = self.sess.run((self.loss, self.optimizer),
                                feed_dict={self.X: X,
                                           self.mutual_info: mutual_info})
        return cost

    def transform(self, x):
        return self.sess.run(self.encoder_op, feed_dict={self.X: x})


def batch_maker(x_train, batch_size, batch_count):
    lower_band = (batch_count * batch_size) % x_train.shape[0]
    upper_band = (batch_count * batch_size + batch_size) % x_train.shape[0]
    x_batch = x_train[lower_band:upper_band, :]
    return x_batch


def main():
    train = pd.read_csv('./data/NSLKDDTrain+/binary.csv', header=None)
    test = pd.read_csv('./data/NSLKDDTest+/binary.csv', header=None)
    x_tr = train.iloc[:, 0:train.shape[1] - 1]
    y_tr = train.iloc[:, train.shape[1] - 1]
    x_te = test.iloc[:, 0:test.shape[1] - 1]
    y_te = test.iloc[:, test.shape[1] - 1]
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    x_tr = min_max_scaler.fit_transform(x_tr)
    x_te = min_max_scaler.fit_transform(x_te)
    mi = mutual_info_weight(x_tr, y_tr)
    gc.collect()
    n_samples = x_tr.shape[0]
    training_epochs = 1000
    batch_size = 250
    display_step = 100
    num_steps = int(training_epochs * n_samples / batch_size)
    gc.collect()
    autoencoder = SupervisedSparseAutoencoder(
        n_input=x_tr.shape[1], n_hidden=30, sample_size=batch_size, rho=.5, reconstruct_reg=1,
        kl_reg=3, regular_reg=(0.00000006 / 2), activation=tf.nn.sigmoid)

    for i in tqdm(range(num_steps)):
        batch_xs = batch_maker(x_tr, batch_size, i)
        l = autoencoder.partial_fit(batch_xs, mi)
        if i % display_step == 0:
            print("mini_batch:", '%04d' % (i + 1), "cost=", l)
    z_test = autoencoder.transform(x_te)
    z_train = autoencoder.transform(x_tr)
    model = binary_svm(z_train, y_tr)


if __name__ == '__main__':
    main()
