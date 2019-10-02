from model.v1.network_template import AbsNeuralNetwork
import tensorflow as tf
import numpy as np


class CaptchaGenerator(AbsNeuralNetwork):
    def build_graph(self):
        with self.graph.as_default():
            gen_x_placeholder = tf.placeholder(tf.float64, shape=(None, *self.feature_shape), name='gen_inputs')
            gen_conv1 = tf.layers.conv2d(inputs=gen_x_placeholder,
                                         filters=32,
                                         kernel_size=[5, 5],
                                         padding='same',
                                         name='gen_conv1',
                                         activation=tf.nn.leaky_relu,
                                         trainable=True)
            normalized_conv1 = self.get_batch_norm(input_tensor=gen_conv1, name='gen_batch_norm1')

            gen_conv2 = tf.layers.conv2d(inputs=normalized_conv1,
                                         filters=32,
                                         kernel_size=[5, 5],
                                         padding='same',
                                         name='gen_conv2',
                                         activation=tf.nn.leaky_relu,
                                         trainable=True)
            normalized_conv2 = self.get_batch_norm(input_tensor=gen_conv2, name='gen_batch_norm2')

            gen_conv3 = tf.layers.conv2d(inputs=normalized_conv2,
                                         filters=3,
                                         kernel_size=[5, 5],
                                         padding='same',
                                         name='gen_conv3',
                                         activation=tf.nn.leaky_relu,
                                         trainable=True)

            normalized_conv2_flat = self.get_batch_norm(input_tensor=gen_conv3, name='gen_batch_norm3', flatten=True)
            output = tf.reshape(normalized_conv2_flat, [-1, *self.feature_shape])
            
            self.tf_nodes['x_placeholder'] = gen_x_placeholder
            self.tf_nodes['gen_conv1'] = gen_conv1
            self.tf_nodes['gen_conv2'] = gen_conv2
            self.tf_nodes['output'] = output
            init = tf.global_variables_initializer()
            self.tf_nodes['init'] = init

    def get_faked_dataset(self, n_sample):
        features = self.generate_captcha(n_sample)
        labels = self.generate_faked_labels(n_sample)
        return features, labels
            
    def generate_captcha(self, n_sample):
        if self.sess is None:
            raise RuntimeError("Train first to generate proper captcha")
            
        with self.graph.as_default():
            rand_inputs = np.random.normal(size=(n_sample, *self.feature_shape))
            rand_inputs = self.reshape_features(rand_inputs)
            # 153600 values, but the requested shape requires a multiple of 28800
            return self.sess.run(self.tf_nodes['output'], feed_dict={self.tf_nodes['x_placeholder']: rand_inputs})

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
