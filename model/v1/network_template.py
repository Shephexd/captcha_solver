import numpy as np
import tensorflow as tf


class AbsNeuralNetwork:
    def __init__(self, graph, sess):
        self.graph = graph
        self.sess = sess
        self.tf_nodes = dict()
        self.build_graph()
        self.data_iter = None

    @property
    def is_runnable(self):
        return bool(self.sess)

    @property
    def feature_shape(self):
        return (60, 160, 3)

    def build_graph(self):
        with self.graph.as_default():
            pass
        
    def calc_cost(self, y_hat, labels):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=labels)
        loss = tf.reduce_sum(cross_entropy, name='loss')
        return loss
    
    def train(self, X, y, learning_rate=0.001, dropout_rate=0.4, epoch=10, batch_size=8, **kwargs):
        with self.graph.as_default():
            pass
        
    def predict(self, X, prob=False):
        assert self.is_runnable, "Model must be trained or loaded"
        with self.graph.as_default():
            pass
        
    def test_model(self, X, y, prob=False):
        assert self.is_runnable, "Model must be trained or loaded"
        with self.graph.as_default():
            pass
        
    def get_accuracy(self, labels, y_hat):
        matched = (labels.argmax(axis=1) == y_hat)
        y_hat.shape[1] == matched.sum(axis=1)
        return labels == y_hat
    
    def reshape_inputs(self, X, y):
        X = self.reshape_features(X)
        y = self.reshape_labels(y)
        return X, y
    
    def reshape_features(self, X):
        if isinstance(X, list):
            X = np.array(X)
            
        if np.ndim(X) == 3:
            X = X.reshape(-1, *self.feature_shape)
        return X
    
    def reshape_labels(self, y):
        if isinstance(y, list):
            y = np.array(y)

        if np.ndim(y) == 2:
            y = y.reshape(-1, 5, 10)
        return y
