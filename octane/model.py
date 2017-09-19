import numpy as np
import tensorflow as tf
from functools import wraps

RANDOM_RANGE = -1.0, 1.0
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_EPOCHS_NUMBER = 1000


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

        self._nn_architecture = []

        shape_len = len(shape)

        self.tf_layers = [self._x]

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

        for layer in self._nn_architecture:
            self.tf_layers.append(
                tf.add(
                    tf.matmul(self.tf_layers[-1], layer['weights']),
                    layer['biases']
                )
            )

        # get last\output layer
        self.output_layer = self.tf_layers[-1]
        self.cost = tf.reduce_mean(
            tf.square(self.output_layer - self._y)
        )

        optimizer = tf.train.GradientDescentOptimizer(
            tf.Variable(self._learning_rate)
        )
        self._train = optimizer.minimize(self.cost)

    def __enter__(self):
        self._session = tf.Session()
        self._session.run(tf.global_variables_initializer())

    def __exit__(self, *args):
        self._session.close()

    # TODO: do smth more clear and not ugly
    def with_session(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            model_instance, *_ = args
            assert isinstance(model_instance, Model) is True
            try:
                model_instance._session
            except AttributeError:
                raise RuntimeError('No session')
            return func(*args, **kwargs)
        return wrapper

    @with_session
    def feed_through(self, data):
        """ feed through the model
        one dimensional data tensor @data """
        sess = self._session
        return sess.run(
            self.output_layer,
            feed_dict={self._x: np.array([data])}
        )

    @with_session
    def train(self, data_set):
        """ trains model with @data_set"""
        sess = self._session
        for epoch in range(self._epochs):
            for ex, ey in zip(*data_set):
                sess.run(
                    self._train,
                    feed_dict={
                        self._x: np.array([ex]),
                        self._y: np.array([ey])
                    }
                )
                # TODO: if debug
                if epoch % 10 == 0 and False:
                    print(
                        sess.run(
                            self.cost,
                            feed_dict={
                                self._x: np.array([ex]),
                                self._y: np.array([ey])
                            }
                        )
                    )
