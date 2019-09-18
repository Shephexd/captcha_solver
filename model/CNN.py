import scipy.misc
import numpy as np
import tensorflow as tf


# TODO Refactor with tf.Estimator
class ConvNet:
    def __init__(self):
        self.graph = tf.Graph()
        self.build_graph()
        self.sess = None
    
    @property
    def is_runnable(self):
        return bool(self.sess)
    
    def build_graph(self):
        with self.graph.as_default():            
            inputs = tf.placeholder(tf.float64, shape=(None, 60, 160, 3), name='inputs')
            labels = tf.placeholder(tf.float64, shape=(None, 5, 10), name='labels')
            
            learning_rate = tf.placeholder(tf.float64, name='learning_rate')
            dropout_rate = tf.placeholder(tf.float64, name='dropout_rate')
            
            conv1 = tf.layers.conv2d(inputs=inputs,
                        filters=32,
                        kernel_size=[5, 5],
                        padding='same',
                        name='conv1',
                        activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name='pool1')
            
            conv2 = tf.layers.conv2d(inputs=pool1,
                        filters=64,
                        kernel_size=[5, 5],
                        padding='same',
                        activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
            pool2_flat = tf.reshape(pool2, [-1, 15 * 40 * 64])
            
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
    
    def train(self, X, y, learning_rate=0.001, dropout_rate=0.4):
        with self.graph.as_default():
            self.sess = tf.Session()

            init = tf.global_variables_initializer()

            x_placeholder = self.graph.get_tensor_by_name('inputs:0')
            y_placeholder = self.graph.get_tensor_by_name('labels:0')
            lr = self.graph.get_tensor_by_name('learning_rate:0')
            dr = self.graph.get_tensor_by_name('dropout_rate:0')

            training_step = cnn.graph.get_operation_by_name('training_step')
            loss = cnn.graph.get_tensor_by_name('loss:0')

            self.sess.run(init)
            for i in range(10):
                _, cost = self.sess.run([training_step, loss], feed_dict={x_placeholder: X, y_placeholder: y, lr: learning_rate, dr: dropout_rate})
                print(cost)

    def predict(self, X, prob=False):
        assert self.is_runnable, "Model must be trained or loaded"
        with self.graph.as_default():
            x_placeholder = self.graph.get_tensor_by_name('inputs:0')
            y_hat = self.graph.get_tensor_by_name('y_hat:0')
            dr = self.graph.get_tensor_by_name('dropout_rate:0')
            output = self.sess.run(tf.nn.softmax(y_hat, axis=2), feed_dict={x_placeholder: X, dr: np.double(0.0)})
            
            if not prob:
                return output.argmax(axis=2)
            else:
                return output


if __name__ == '__main__':
    # X:(N, 60, 160, 3)
    # Y:(N, 5, 10)
    input_data = scipy.misc.imread('42119.png')
    inputs = inputs = input_data.reshape(-1, 60, 160, 3)/255
    label = np.eye(10)[np.array([int(i) for i in '42119'])]
    labels = label.reshape(-1, 5, 10)

    cnn = ConvNet()
    cnn.train(inputs, labels)
    y_hat = cnn.predict(inputs, prob=True)
    print(y_hat)
