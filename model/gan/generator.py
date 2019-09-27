from model.network_template import AbsNeuralNetwork
import tensorflow as tf
import numpy as np


class CaptchaGenerator(AbsNeuralNetwork):
    def build_graph(self):
        with self.graph.as_default():
            gen_x_placeholder = tf.placeholder(tf.float64, shape=(None, 60, 160, 3), name='gen_inputs')
            gen_conv1 = tf.layers.conv2d(inputs=gen_x_placeholder,
                                         filters=32,
                                         kernel_size=[4, 4],
                                         padding='same',
                                         name='conv1',
                                         activation=tf.nn.relu)
            gen_conv2 = tf.layers.conv2d(inputs=gen_conv1,
                                         filters=3,
                                         kernel_size=[5, 5],
                                         padding='same',
                                         activation=tf.nn.relu)
            
            self.tf_nodes['x_placeholder'] = gen_x_placeholder
            self.tf_nodes['gen_conv1'] = gen_conv1
            self.tf_nodes['gen_conv2'] = gen_conv2

            init = tf.global_variables_initializer()
            self.sess.run(init)

    def get_faked_dataset(self, n_sample):
        features = self.generate_captcha(n_sample)
        labels = self.generate_faked_labels(n_sample)
        return features, labels
            
    def generate_captcha(self, n_sample):
        if self.sess is None:
            raise RuntimeError("Train first to generate proper captcha")
            
        with self.graph.as_default():
            rand_inputs = np.random.rand(n_sample, 60, 160, 3)
            rand_inputs = self.reshape_features(rand_inputs)
            return self.sess.run(self.tf_nodes['gen_conv2'], feed_dict={self.tf_nodes['x_placeholder']: rand_inputs})

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
    print(generator.get_faked_dataset(3))