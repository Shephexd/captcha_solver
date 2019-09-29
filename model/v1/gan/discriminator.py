import tensorflow as tf
import numpy as np
from model.v1.network_template import AbsNeuralNetwork


class CaptchaDiscriminator(AbsNeuralNetwork):
    def build_discriminator(self, features, dropout_rate, learning_rate):
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
                                 name='disc_conv2',
                                 activation=tf.nn.relu,
                                 reuse=tf.AUTO_REUSE)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[5, 5], strides=5, name='disc_pool2')
        pool2_flat = tf.reshape(pool2, [-1, np.prod([*self.feature_shape])])

        batch_normal = tf.layers.batch_normalization(pool2_flat, reuse=tf.AUTO_REUSE, name='disc_pool2')
        dense = tf.layers.dense(inputs=batch_normal, units=1024, activation=tf.nn.relu, name='disc_fc', reuse=tf.AUTO_REUSE)
        dropout = tf.layers.dropout(inputs=dense, rate=dropout_rate, name='disc_dropout', training=True)

        logits = tf.layers.dense(inputs=dropout, units=2, name='disc_logits', reuse=tf.AUTO_REUSE)
        outputs = tf.nn.softmax(logits)

        self.tf_nodes['learning_rate'] = learning_rate
        self.tf_nodes['dropout_rate'] = dropout_rate
        self.tf_nodes['conv1'] = conv1
        self.tf_nodes['pool1'] = pool1
        self.tf_nodes['conv2'] = conv2
        self.tf_nodes['pool2'] = pool2
        self.tf_nodes['dense'] = dense
        self.tf_nodes['logits'] = logits

        return outputs


if __name__ == '__main__':
    from preprocess.load_data import get_data

    x, y = get_data(pardir='/Users/shephexd/Documents/github/captcha_solver/data')
    print(x, y)

    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    discriminator = CaptchaDiscriminator(graph=graph, sess=sess)

    with graph.as_default():
        input_real = tf.placeholder(tf.float64, shape=(None, *discriminator.feature_shape), name='input_real')
        input_fake = tf.placeholder(tf.float64, shape=(None, *discriminator.feature_shape), name='input_fake')
        dropout_rate = tf.placeholder(tf.float64, name='dropout_rate')

        logit_real = discriminator.build_discriminator(input_real, dropout_rate)
        logit_fake = discriminator.build_discriminator(input_fake, dropout_rate)
        init = tf.global_variables_initializer()

        sess.run(init)
        sess.run(logit_real, feed_dict={input_real: x, dropout_rate: 0.5})
        sess.run(logit_fake, feed_dict={input_fake: x, dropout_rate: 0.5})
