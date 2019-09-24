import tensorflow as tf


class AbsNeuralNetwork:
    def __init__(self):
        self.graph = tf.Graph()
        self.build_graph()
        self.sess = None
        self.data_iter = None

    @property
    def is_runnable(self):
        return bool(self.sess)

    def build_graph(self):
        with self.graph.as_default():
            pass
    
    def calc_cost(self, y_hat, labels):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=labels)
        loss = tf.reduce_sum(cross_entropy, name='loss')
        return loss
    
    def train(self, X, y, learning_rate=0.001, dropout_rate=0.4, epoch=10, batch_size=8):
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
        return X
    
    def reshape_labels(self, y):
        return y
