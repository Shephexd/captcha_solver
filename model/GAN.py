from .network_template import AbsNeuralNetwork
import tensorflow as tf
import numpy as np


class GenAdvNetwork(AbsNeuralNetwork):
    def build_graph(self):
        with self.graph.as_default():
            self.build_generator()
            self.build_discriminator()
            
    def build_generator(self):
        batch_size_placeholder = tf.placeholder(tf.int64, name='batch_size')

        gen_x_placeholder = tf.placeholder(tf.float64, shape=(None, 60, 160, 3), name='gen_inputs')
        self.gen_conv1 = tf.layers.conv2d(inputs=gen_x_placeholder,
                    filters=32,
                    kernel_size=[4, 4],
                    padding='same',
                    name='conv1',
                    activation=tf.nn.relu)
        
        self.gen_conv2 = tf.layers.conv2d(inputs=self.gen_conv1,
                                          filters=3,
                                          kernel_size=[5, 5],
                                          padding='same',
                                          activation=tf.nn.relu)
        
    def build_discriminator(self):
        with self.graph.as_default():
            batch_size_placeholder = tf.placeholder(tf.int64, name='batch_size')

            x_placeholder = tf.placeholder(tf.float64, shape=(None, 60, 160, 3), name='disc_inputs')
            y_placeholder = tf.placeholder(tf.double, shape=(None, 5, 10), name='disc_labels')
            
            train_dataset = tf.data.Dataset.from_tensor_slices(tensors=(x_placeholder, y_placeholder))
            train_batch_dataset = train_dataset.batch(batch_size_placeholder).repeat()
            train_dataset = tf.data.Dataset.from_tensor_slices(tensors=(x_placeholder, y_placeholder))
            test_batch_dataset = train_dataset.batch(batch_size_placeholder)
            
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
            loss = self.calc_cost(y_hat, labels)
            
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            training_step = optimizer.minimize(loss=loss, name='training_step')
    
    def get_faked_dataset(self, n_sample):
        features = self.generate_captcha(n_sample)
        labels = self.generate_faked_labels(n_sample)
        return features, labels
    
    def get_genuine_dataset(self, X):
        features = self.reshape_features(X)
        labels = self.generate_genuine_labels(n_sample)
        return features, labels
    
    def generate_captcha(self, n_sample):
        if self.sess is None:
            raise RuntimeError("Train first to generate proper captcha")
            
        with self.graph.as_default():
            x_placeholder = gan.graph.get_tensor_by_name('gen_inputs:0')
            rand_inputs = np.random.rand(n_sample, 60, 160, 3)
            rand_inputs = self.reshape_features(rand_inputs)
            return self.sess.run(self.gen_conv2, feed_dict={x_placeholder: rand_inputs})

    def generate_faked_labels(n_sample):
        return self.generate_labels(n_sample, label_idx=0)

    def generate_genuine_labels(n_sample):
        return self.generate_labels(n_sample, label_idx=1)
    
    def generate_labels(n_sample, label_idx):
        target = np.ones([n_sample, label_idx], dtype=np.int).reshape(-1)
        labels = np.eye(2)[target]
        return labels
    
    def train(self, X, batch_size):
        with self.graph.as_default():
            X = self.reshape_features(X)
            n_batch = X.shape[0] / batch_size
            self.sess = tf.Session(graph=self.graph)
            
            gen_features, gen_labels = self.get_genuine_dataset(X)
            
            init = tf.global_variables_initializer()
            self.sess.run(init)
