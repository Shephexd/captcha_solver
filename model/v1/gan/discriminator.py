import tensorflow as tf
from model.v1.network_template import AbsNeuralNetwork


class CaptchaDiscriminator(AbsNeuralNetwork):
    def build_discriminator(self, features):
        learning_rate = tf.placeholder(tf.float64, name='learning_rate')
        dropout_rate = tf.placeholder(tf.float64, name='dropout_rate')

        conv1 = tf.layers.conv2d(inputs=features,
                                 filters=32,
                                 kernel_size=[5, 5],
                                 padding='same',
                                 name='disc_conv1',
                                 activation=tf.nn.relu,
                                 reuse=tf.AUTO_REUSE)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[4, 4], strides=4, name='disc_pool1')

        conv2 = tf.layers.conv2d(inputs=pool1,
                                 filters=64,
                                 kernel_size=[5, 5],
                                 padding='same',
                                 activation=tf.nn.relu,
                                 reuse=tf.AUTO_REUSE)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[5, 5], strides=5)
        pool2_flat = tf.reshape(pool2, [-1, 3 * 8 * 64])

        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu, reuse=tf.AUTO_REUSE)
        dropout = tf.layers.dropout(inputs=dense, rate=dropout_rate, training=True)

        logits = tf.layers.dense(inputs=dropout, units=1, )
        self.tf_nodes['learning_rate'] = learning_rate
        self.tf_nodes['dropout_rate'] = dropout_rate
        self.tf_nodes['conv1'] = conv1
        self.tf_nodes['pool1'] = pool1
        self.tf_nodes['conv2'] = conv2
        self.tf_nodes['pool2'] = pool2
        self.tf_nodes['dense'] = dense
        self.tf_nodes['logits'] = logits

        return logits

    def train(self, X, batch_size):
        with self.graph.as_default():
            X = self.reshape_features(X)
            n_batch = X.shape[0] / batch_size

            x_placeholder = self.graph.get_tensor_by_name('inputs:0')
            y_placeholder = self.graph.get_tensor_by_name('labels:0')

            bs = self.graph.get_tensor_by_name('batch_size:0')
            lr = self.graph.get_tensor_by_name('learning_rate:0')
            dr = self.graph.get_tensor_by_name('dropout_rate:0')

            training_step = cnn.graph.get_operation_by_name('training_step')
            loss = cnn.graph.get_tensor_by_name('loss:0')


