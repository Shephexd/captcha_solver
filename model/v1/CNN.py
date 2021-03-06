import scipy.misc
import numpy as np
import tensorflow as tf


# TODO Refactor with tf.Estimator
class ConvNet:
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
            batch_size_placeholder = tf.placeholder(tf.int64, name='batch_size')

            x_placeholder = tf.placeholder(tf.float64, shape=(None, 60, 160, 3), name='inputs')
            y_placeholder = tf.placeholder(tf.double, shape=(None, 5, 10), name='labels')
            
            train_dataset = tf.data.Dataset.from_tensor_slices(tensors=(x_placeholder, y_placeholder))
            train_batch_dataset = train_dataset.batch(batch_size_placeholder).repeat()
            test_dataset = tf.data.Dataset.from_tensor_slices(tensors=(x_placeholder, y_placeholder))
            test_batch_dataset = test_dataset.batch(batch_size_placeholder)
            
            iterator = tf.data.Iterator.from_structure(train_batch_dataset.output_types,
                                                       train_batch_dataset.output_shapes)
            feature, labels = iterator.get_next()
            
            self.train_iter_init_op = iterator.make_initializer(train_batch_dataset)
            self.test_iter_init_op = iterator.make_initializer(test_batch_dataset)

            learning_rate = tf.placeholder(tf.float64, name='learning_rate')
            dropout_rate = tf.placeholder(tf.float64, name='dropout_rate')
            
            conv1 = tf.layers.conv2d(inputs=feature,
                        filters=32,
                        kernel_size=[5, 5],
                        padding='same',
                        name='conv1',
                        activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[4, 4], strides=4, name='pool1')
            
            conv2 = tf.layers.conv2d(inputs=pool1,
                        filters=64,
                        kernel_size=[5, 5],
                        padding='same',
                        activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[5, 5], strides=5)
            pool2_flat = tf.reshape(pool2, [-1, 3 * 8 * 64])
            
            dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
            dropout = tf.layers.dropout(inputs=dense, rate=dropout_rate, training=True)
            
            logits = tf.layers.dense(inputs=dropout, units=50)
            y_hat = tf.reshape(logits, [-1, 5, 10], name='y_hat')
            
            loss = self.calc_cost(y_hat, labels)
            
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            training_step = optimizer.minimize(loss=loss, name='training_step')
            
    def calc_cost(self, y_hat, labels):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=labels)
        loss = tf.reduce_sum(cross_entropy, name='loss')
        return loss
    
    def train(self, X, y, learning_rate=0.001, dropout_rate=0.4, epoch=10, batch_size=8):
        with self.graph.as_default():
            n_batch = X.shape[0] / batch_size
            self.sess = tf.Session()
            
            x_placeholder = self.graph.get_tensor_by_name('inputs:0')
            y_placeholder = self.graph.get_tensor_by_name('labels:0')

            init = tf.global_variables_initializer()
            bs = self.graph.get_tensor_by_name('batch_size:0')
            lr = self.graph.get_tensor_by_name('learning_rate:0')
            dr = self.graph.get_tensor_by_name('dropout_rate:0')
            
            training_step = cnn.graph.get_operation_by_name('training_step')
            loss = cnn.graph.get_tensor_by_name('loss:0')

            self.sess.run(init)
            self.sess.run(self.train_iter_init_op, 
                          feed_dict={
                              x_placeholder: X, 
                              y_placeholder: y, 
                              bs: batch_size})
            
            for i in range(epoch):
                tot_cost = 0
                for n in range(int(n_batch)):
                    _, cost = self.sess.run([training_step, loss], 
                                            feed_dict={
                                                lr: learning_rate, 
                                                dr: dropout_rate
                                            })
                    tot_cost += cost
                    print("epoch:%d, batch: %d" % (i+1, n+1))
                print("COST:", tot_cost/n_batch)
    
    def predict(self, X, prob=False):
        assert self.is_runnable, "Model must be trained or loaded"
        with self.graph.as_default():
            X = self.reshape_features(X)
            y = np.zeros([X.shape[0], 5, 10])

            x_placeholder = self.graph.get_tensor_by_name('inputs:0')
            y_placeholder = self.graph.get_tensor_by_name('labels:0')
            bs = self.graph.get_tensor_by_name('batch_size:0')
            self.sess.run(self.test_iter_init_op, 
                          feed_dict={
                              x_placeholder: X, 
                              y_placeholder: y,
                              bs: 1})
            
            y_hat = self.graph.get_tensor_by_name('y_hat:0')
            dr = self.graph.get_tensor_by_name('dropout_rate:0')
            output = self.sess.run(y_hat, feed_dict={dr: np.double(0.0)})
            if not prob:
                return output.argmax(axis=2)
            else:
                return output
    
    def test_model(self, X, y, prob=False):
        assert self.is_runnable, "Model must be trained or loaded"
        with self.graph.as_default():
            X, y = self.reshape_inputs(X, y)
            x_placeholder = self.graph.get_tensor_by_name('inputs:0')
            y_placeholder = self.graph.get_tensor_by_name('labels:0')
            bs = self.graph.get_tensor_by_name('batch_size:0')
            self.sess.run(self.test_iter_init_op, 
                          feed_dict={
                              x_placeholder: X, 
                              y_placeholder: y,
                              bs: 1})

            y_hat = self.graph.get_tensor_by_name('y_hat:0')
            dr = self.graph.get_tensor_by_name('dropout_rate:0')
            output = self.sess.run(tf.nn.softmax(y_hat, axis=2), feed_dict={x_placeholder: X, dr: np.double(0.0)})
            
            if not prob:
                return output.argmax(axis=2)
            else:
                return output
    
    def get_accuracy(self, labels, y_hat):
        matched = (labels.argmax(axis=1) == y_hat)
        y_hat.shape[1] == matched.sum(axis=1)
        return labels == y_hat
    
    def reshape_inputs(self, X, y):
        X = self.reshape_features(X)
        y = self.reshape_labels(y)
        return X, y
    
    def reshape_features(self, X):
        if np.ndim(X) == 3:
            X = X.reshape(-1, 60, 160, 3)
        return X
    
    def reshape_labels(self, y):
        if np.ndim(y) == 2:
            y = y.reshape(-1, 5, 10)
        return y


if __name__ == '__main__':
    # X:(N, 60, 160, 3)
    # Y:(N, 5, 10)
    input_data = scipy.misc.imread('42119.png')
    inputs = inputs = input_data.reshape(-1, 60, 160, 3)/255
    label = np.eye(10)[np.array([int(i) for i in '42119'])]
    labels = label.reshape(-1, 5, 10)

    cnn = ConvNet()
    cnn.train(inputs, labels, epoch=200, batch_size=128)
    
    SAMPLE_IDX = 0
    print(y[SAMPLE_IDX].argmax(axis=1))
    print(cnn.test_model(inputs[SAMPLE_IDX].reshape(-1, 60, 160, 3), inputs[SAMPLE_IDX].reshape(-1, 5, 10)))
