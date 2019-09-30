from model.v1.network_template import AbsNeuralNetwork
from model.v1.gan.generator import CaptchaGenerator
from model.v1.gan.discriminator import CaptchaDiscriminator
import tensorflow as tf
from matplotlib import pyplot as plt


class CaptchaGenAdvNet(AbsNeuralNetwork):
    def __init__(self, graph, sess):
        super().__init__(graph=graph, sess=sess)
        self.generator = CaptchaGenerator(graph, sess)
        self.discriminator = CaptchaDiscriminator(graph, sess)

    def train(self, X, learning_rate=0.001, dropout_rate=0.1, epoch=1, batch_size=8, **kwargs):
        X = self.reshape_features(X)
        n_batch = X.shape[0] / batch_size

        # Training Variables for each optimizer
        # By default in TensorFlow, all variables are updated by each optimizer, so we
        # need to precise for each one of them the specific variables to update.
        # Generator Network Variables

        with self.graph.as_default():
            input_real = tf.placeholder(tf.float64, shape=(None, 60, 160, 3), name='input_real')
            input_fake = self.generator.tf_nodes['gen_conv2']

            learning_rate_placeholder = tf.placeholder(tf.float64, name='learning_rate')
            dropout_rate_placeholder = tf.placeholder(tf.float64, name='dropout_rate')

            output_real = self.discriminator.build_discriminator(input_real, dropout_rate_placeholder,
                                                                 learning_rate_placeholder)
            output_fake = self.discriminator.build_discriminator(input_fake, dropout_rate_placeholder,
                                                                 learning_rate_placeholder)
            init = tf.global_variables_initializer()

            self.sess.run(init)

            gen_loss = -tf.reduce_mean(tf.log(output_fake))
            disc_loss = -tf.reduce_mean(tf.log(output_real) + tf.log(1. - output_fake))

            gen_vars = [node for node in graph._collections['trainable_variables'] if node.name.startswith('gen')]
            disc_vars = [node for node in graph._collections['trainable_variables'] if node.name.startswith('disc')]

            optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)
            optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)

            train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
            train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

            init = tf.global_variables_initializer()
            self.sess.run(init)

            for i in range(epoch):
                _, gen_cost, output_f = self.sess.run([train_gen, gen_loss, output_fake],
                                                      feed_dict={
                                                          input_real: X,
                                                          self.generator.tf_nodes[
                                                              'x_placeholder']: self.generator.generate_captcha(
                                                              X.shape[0]),
                                                          learning_rate_placeholder: learning_rate,
                                                          dropout_rate_placeholder: dropout_rate
                                                      })
                _, disc_cost, output_r = self.sess.run([train_disc, gen_loss, output_real],
                                                       feed_dict={
                                                           input_real: X,
                                                           self.generator.tf_nodes[
                                                               'x_placeholder']: self.generator.generate_captcha(
                                                               X.shape[0]),
                                                           learning_rate_placeholder: learning_rate,
                                                           dropout_rate_placeholder: dropout_rate
                                                       })

                if i % 10 == 0:
                    print(output_f[:3], output_r[:3])
                    print(gen_cost, disc_cost)
                    sample = self.generator.generate_captcha(1)
                    plt.imshow(sample.reshape(60, 160, 3))
                    plt.show()


if __name__ == '__main__':
    from preprocess.load_data import get_data

    x, y = get_data(pardir='/home/fount/poc_server/notebooks/Captcha/data')
    print(x, y)

    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    gan = CaptchaGenAdvNet(graph=graph, sess=sess)
    gan.train(X=x[:50], epoch=60, learning_rate=0.0002)
    print(gan)
