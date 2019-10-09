from model.v1.network_template import AbsNeuralNetwork
import tensorflow as tf
import numpy as np


class CaptchaGenerator(AbsNeuralNetwork):
    def build_graph(self):
        with self.graph.as_default():
            # 100 -> 4*4 * 1024 -> 8*8 * 512 -> 16 * 16 * 256 -> 32 *32 128 -> 64 * 64 * 3
            # 100 -> 3 * 8 * 256 -> 15 * 40 * 128 -> 30 * 80 * 64 -> 60 * 160 * 3
            gen_x_placeholder = tf.placeholder(tf.float64, shape=(None, 100), name='gen_inputs')

            input_dense = tf.layers.dense(inputs=gen_x_placeholder, units=192,
                                          activation=tf.nn.relu, name='gen_input_dense',
                                          reuse=tf.AUTO_REUSE)

            input_dense_norm = self.get_batch_norm(input_tensor=input_dense, name='gen_input_dense_norm1', flatten=True)
            input_conv = tf.reshape(input_dense_norm, [-1, 3, 8, 8])

            gen_conv1 = tf.layers.conv2d(inputs=input_conv,
                                         filters=64,
                                         kernel_size=(4, 4),
                                         padding='same',
                                         name='gen_conv1',
                                         activation=tf.nn.relu,
                                         trainable=True)

            normalized_conv1 = self.get_batch_norm(input_tensor=gen_conv1, name='gen_batch_norm1')
            gen_conv2_input = tf.reshape(normalized_conv1, [-1, 6, 16, 16])

            gen_conv2 = tf.layers.conv2d(inputs=gen_conv2_input,
                                         filters=32,
                                         kernel_size=(6, 6),
                                         padding='same',
                                         name='gen_conv2',
                                         activation=tf.nn.relu,
                                         trainable=True)
            normalized_conv2 = self.get_batch_norm(input_tensor=gen_conv2, name='gen_batch_norm2')
            gen_conv3_input = tf.reshape(normalized_conv2, [-1, 12, 32, 8])

            gen_conv3 = tf.layers.conv2d(inputs=gen_conv3_input,
                                         filters=50,
                                         kernel_size=(8, 8),
                                         padding='same',
                                         name='gen_conv3',
                                         activation=tf.nn.relu,
                                         trainable=True)
            normalized_conv3 = self.get_batch_norm(input_tensor=gen_conv3, name='gen_batch_norm3')
            gen_conv4_input = tf.reshape(normalized_conv3, [-1, 60, 160, 2])

            gen_conv4 = tf.layers.conv2d(inputs=gen_conv4_input,
                                         filters=3,
                                         kernel_size=(12, 12),
                                         padding='same',
                                         name='gen_conv4',
                                         activation=tf.nn.tanh,
                                         trainable=True)

            reshaped_output = tf.reshape(gen_conv4, [-1, *self.feature_shape])

            self.tf_nodes['x_placeholder'] = gen_x_placeholder
            self.tf_nodes['gen_conv1'] = gen_conv1
            self.tf_nodes['gen_conv2'] = gen_conv2
            self.tf_nodes['output'] = reshaped_output
            init = tf.global_variables_initializer()
            self.tf_nodes['init'] = init

    def get_faked_dataset(self, n_sample):
        features = self.generate_captcha(n_sample)
        labels = self.generate_faked_labels(n_sample)
        return features, labels

    def generate_radom_noise(self, n_samples):
        return np.random.normal(size=(n_samples, 100))

    def generate_captcha(self, n_samples):
        if self.sess is None:
            raise RuntimeError("Train first to generate proper captcha")
            
        with self.graph.as_default():
            rand_inputs = self.generate_radom_noise(n_samples=n_samples)
            output = self.sess.run(self.tf_nodes['output'], feed_dict={self.tf_nodes['x_placeholder']: rand_inputs})
            return (output + 1) / 2

    def generate_faked_labels(self, n_sample):
        return self.generate_labels(n_sample, label_idx=0)

    def generate_genuine_labels(self, n_sample):
        return self.generate_labels(n_sample, label_idx=1)

    @staticmethod
    def generate_labels(n_sample, label_idx):
        target = np.ones([n_sample, label_idx], dtype=np.int).reshape(-1)
        labels = np.eye(2)[target]
        return labels


if __name__ == '__main__':
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    generator = CaptchaGenerator(graph=graph, sess=sess)
    sess.run(generator.tf_nodes['init'])
    generated = generator.generate_captcha(4)
    print(generated)
    print(generated.shape)
