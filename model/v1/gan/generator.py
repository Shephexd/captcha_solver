from model.v1.network_template import AbsNeuralNetwork
import tensorflow as tf
import numpy as np


class CaptchaGenerator(AbsNeuralNetwork):
    def build_graph(self):
        with self.graph.as_default():
            gen_x_placeholder = tf.placeholder(tf.float64, shape=(None, *self.feature_shape), name='gen_inputs')
            gen_conv1 = tf.layers.conv2d(inputs=gen_x_placeholder,
                                         filters=4,
                                         kernel_size=[4, 4],
                                         padding='same',
                                         name='gen_conv1',
                                         activation=tf.nn.relu,
                                         trainable=True)
            gen_conv2 = tf.layers.conv2d(inputs=gen_conv1,
                                         filters=3,
                                         kernel_size=[5, 5],
                                         padding='same',
                                         name='gen_conv2',
                                         activation=tf.nn.relu,
                                         trainable=True)

            conv2_flat = tf.reshape(gen_conv2, [-1, np.prod([*self.feature_shape])])
            min_v = tf.reduce_min(conv2_flat)
            max_v = tf.reduce_max(conv2_flat)
            normalized_flat = (conv2_flat - min_v) / (max_v - min_v)
            output = tf.reshape(normalized_flat, [-1, *self.feature_shape])
            
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
