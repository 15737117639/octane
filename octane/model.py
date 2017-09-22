import sys
import logging
import random

import numpy as np
import tensorflow as tf

RANDOM_RANGE = -1.0, 1.0
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_EPOCHS_NUMBER = 1000

model_logger = logging.getLogger('Model')
model_logger.setLevel(logging.DEBUG)
logging.info('Model loaded')


class ModelException(Exception):
    """ define our ModelException class """
    pass


class Model:
    """ a neural network model"""

    def __init__(self, shape,
                 learning_rate=DEFAULT_LEARNING_RATE,
                 epochs=DEFAULT_EPOCHS_NUMBER):
        """ create a model with shape @shape
        e.g [2, 3, 1] will create a model with
        2 input neurons 3 hidden-layer neurons
        and 1 output neuron """

        self._shape = shape
        self._learning_rate = learning_rate
        self._epochs = epochs

        # create placeholders for input
        # and output data
        self._x = tf.placeholder(tf.float32)
        self._y = tf.placeholder(tf.float32)

        # init session as None
        self._session = None

        # here we would have our model
        # architecture
        self._nn_architecture = []

        # a list of layers
        self._tf_layers = [self._x]

        shape_len = len(shape)

        # generate layers
        for i in range(shape_len - 1):
            # iterate through each pair in @shape
            out_prev_layer, in_next_layer, *_ = shape[i:]
            layer_weights = tf.Variable(
                tf.random_uniform(
                    [out_prev_layer, in_next_layer],
                    *RANDOM_RANGE))
            layer_biases = tf.Variable(
                tf.random_uniform(
                    [in_next_layer],
                    *RANDOM_RANGE)
            )
            layer = {
                'weights': layer_weights,
                'biases': layer_biases
            }
            self._nn_architecture.append(layer)

        # create computational graph
        for layer in self._nn_architecture:
            self._tf_layers.append(
                tf.add(
                    tf.matmul(self._tf_layers[-1], layer['weights']),
                    layer['biases']
                )
            )

        # get last\output layer
        self._output_layer = self._tf_layers[-1]

        # minimize (data - prediction)**2
        self._cost = tf.reduce_mean(
            tf.square(self._output_layer - self._y)
        )

        self._optimizer = tf.train.GradientDescentOptimizer(
            tf.Variable(self._learning_rate)
        )
        self._train = self._optimizer.minimize(self._cost)

        # saver
        self._saver = tf.train.Saver()

    def __enter__(self):
        """ use Model object with context operator """
        self._session = tf.Session()
        self._session.run(tf.global_variables_initializer())

    def __exit__(self, *args):
        self._session.close()

    def save(self, path_to_file):
        """ saves the model """
        if not self._with_session():
            raise ModelException("No session")
        self._saver.save(self._session, path_to_file)
        logging.info('model saved to f{path_to_file}')

    def restore(self, path_to_file):
        """ restoring saved model """
        if not self._with_session():
            raise ModelException("No session")
        self._saver.restore(self._session, path_to_file)
        logging.info('model restored from f{path_to_file}')

    # TODO: do smth more clear and not ugly
    def _with_session(self):
        return self._session is not None

    def feed_through(self, data):
        """ feed through the model
        one dimensional data tensor @data """
        if not self._with_session():
            raise ModelException("No session")
        sess = self._session
        return sess.run(
            self._output_layer,
            feed_dict={self._x: np.array([data])}
        )

    def fit_model(self, data_set):
        """ trains model with @data_set"""
        if not self._with_session():
            raise ModelException("No session")
        sess = self._session
        model_logger.info('Starting training model')
        data = list(zip(*data_set))
        tf.summary.scalar('total loss', self._cost)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(
                'train', # TODO: os.path.join
                sess._graph
        )
        for epoch in range(self._epochs):
            random.shuffle(data)
            epoch_loss = 0
            for ex, ey in data:
                _, cost, mergd = sess.run(
                    [self._train, self._cost, merged],
                    feed_dict={
                        self._x: np.array([ex]),
                        self._y: np.array([ey])
                    }
                )
                epoch_loss += cost
            epoch_loss /= len(data)
            
            train_writer.add_summary(mergd, epoch)
            model_logger.info(f'Epoch {epoch}, RMS epoch loss: {epoch_loss**0.5}')
