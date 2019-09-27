from model.v1.network_template import AbsNeuralNetwork
from model.v1.gan.generator import CaptchaGenerator
from model.v1.gan.discriminator import CaptchaDiscriminator
import tensorflow as tf


class CaptchaGenAdvNet(AbsNeuralNetwork):
    def __init__(self, graph, sess):
        super().__init__(graph=graph, sess=sess)
        self.generator = CaptchaGenerator(graph, sess)
        self.discriminator = CaptchaDiscriminator(graph, sess)

    def train(self, X, learning_rate=0.001, dropout_rate=0.4, epoch=1, batch_size=8, **kwargs):
        X = self.reshape_features(X)
        n_batch = X.shape[0] / batch_size

        input_real = tf.placeholder(tf.float64, shape=(None, 60, 160, 3), name='input_real')
        input_fake = tf.placeholder(tf.float64, shape=(None, 60, 160, 3), name='input_fake')

        disc_real = self.discriminator.build_discriminator(input_real)
        disc_fake = self.discriminator.build_discriminator(input_fake)

        # Build Loss
        gen_loss = -tf.reduce_mean(tf.log(disc_fake))
        disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))

        optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)
        optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # Training Variables for each optimizer
        # By default in TensorFlow, all variables are updated by each optimizer, so we
        # need to precise for each one of them the specific variables to update.
        # Generator Network Variables

        gen_vars = [self.generator.tf_nodes['gen_conv1'], self.generator.tf_nodes['gen_conv2']]

        # Discriminator Network Variables
        disc_vars = [
            self.discriminator.tf_nodes['conv1'], self.discriminator.tf_nodes['pool1'],
            self.discriminator.tf_nodes['conv2'], self.discriminator.tf_nodes['pool2'],
            self.discriminator.tf_nodes['dense'], self.discriminator.tf_nodes['logits']
        ]

        # Create training operations
        train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
        train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        for i in range(epoch):
            _, _, gen_cost, disc_cost = self.sess.run([train_gen, train_disc, gen_loss, disc_loss],
                                                      feed_dict={
                                                          input_real: X,
                                                          disc_fake: self.generator.generate_captcha(X.shape[0]),
                                                          self.discriminator.tf_nodes['learning_rate']: learning_rate,
                                                          self.discriminator.tf_nodes['dropout_rate']: dropout_rate
                                                      })


if __name__ == '__main__':
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    gan = CaptchaGenAdvNet(graph=graph, sess=sess)

