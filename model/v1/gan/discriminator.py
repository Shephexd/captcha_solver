import tensorflow as tf
from model.v1.network_template import AbsNeuralNetwork


class CaptchaDiscriminator(AbsNeuralNetwork):
    def build_discriminator(self, features, dropout_rate, learning_rate):
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            conv1 = tf.layers.conv2d(inputs=features,
                                     filters=128,
                                     kernel_size=[4, 4],
                                     padding='same',
                                     name='disc_conv1',
                                     activation=tf.nn.leaky_relu,
                                     reuse=tf.AUTO_REUSE)
            #pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[4, 4], strides=2, name='disc_pool1')
            conv1 = tf.layers.dropout(inputs=conv1, rate=dropout_rate, name='disc_dropout1')
            normalized_pool1 = self.get_batch_norm(input_tensor=conv1, name='disc_norm_pool1', reuse=tf.AUTO_REUSE)

            conv2 = tf.layers.conv2d(inputs=normalized_pool1,
                                     filters=64,
                                     kernel_size=[5, 5],
                                     padding='same',
                                     name='disc_conv2',
                                     activation=tf.nn.leaky_relu,
                                     reuse=tf.AUTO_REUSE)
            #pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[5, 5], strides=2, name='disc_pool2', padding='same')
            conv2 = tf.layers.dropout(inputs=conv2, rate=dropout_rate, name='disc_dropout2')
            normalized_pool2 = self.get_batch_norm(input_tensor=conv2, name='disc_norm_pool2', reuse=tf.AUTO_REUSE)

            conv3 = tf.layers.conv2d(inputs=normalized_pool2,
                                     filters=32,
                                     kernel_size=[8, 8],
                                     padding='same',
                                     name='disc_conv3',
                                     activation=tf.nn.leaky_relu,
                                     reuse=tf.AUTO_REUSE)

            # pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[8, 8], strides=2, name='disc_pool3', padding='same')
            conv3 = tf.layers.dropout(inputs=conv3, rate=dropout_rate, name='disc_dropout3')
            normalized_pool3 = self.get_batch_norm(input_tensor=conv3, name='disc_norm_pool3', flatten=True, reuse=tf.AUTO_REUSE)

            dense = tf.layers.dense(inputs=normalized_pool3, units=128, activation=tf.nn.relu, name='disc_fc', reuse=tf.AUTO_REUSE)
            logits = tf.layers.dense(inputs=dense, units=2, name='disc_logits', reuse=tf.AUTO_REUSE)
            outputs = tf.nn.softmax(logits)

            self.tf_nodes['learning_rate'] = learning_rate
            self.tf_nodes['dropout_rate'] = dropout_rate
            self.tf_nodes['conv1'] = conv1
            self.tf_nodes['conv2'] = conv2
            self.tf_nodes['conv3'] = conv3
            self.tf_nodes['dense'] = dense
            self.tf_nodes['logits'] = logits

            return outputs


if __name__ == '__main__':
    from preprocess.load_data import get_data
    import numpy as np
    #x, y = get_data(pardir='/Users/shephexd/Documents/github/captcha_solver/data')
    #print(x, y)
    x = np.random.randn(3, 60, 160, 3)

    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    discriminator = CaptchaDiscriminator(graph=graph, sess=sess)

    with graph.as_default():
        input_real = tf.placeholder(tf.float64, shape=(None, *discriminator.feature_shape), name='input_real')
        input_fake = tf.placeholder(tf.float64, shape=(None, *discriminator.feature_shape), name='input_fake')
        dropout_rate = tf.placeholder(tf.float64, name='dropout_rate')
        learning_rate = tf.placeholder(tf.float64, name='dropout_rate')

        logit_real = discriminator.build_discriminator(input_real, dropout_rate, learning_rate)
        logit_fake = discriminator.build_discriminator(input_fake, dropout_rate, learning_rate)
        init = tf.global_variables_initializer()

        sess.run(init)
        print(sess.run(logit_real, feed_dict={input_real: x, dropout_rate: 0.5, learning_rate: 0.01}))
        print(sess.run(logit_fake, feed_dict={input_fake: x, dropout_rate: 0.5, learning_rate: 0.01}))
