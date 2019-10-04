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

    def train(self, X, learning_rate=0.001, dropout_rate=0.1, epoch=1, batch_size=8, decay_rate=0.96, **kwargs):
        X = self.reshape_features(X)
        n_batch = X.shape[0] / batch_size

        # Training Variables for each optimizer
        # By default in TensorFlow, all variables are updated by each optimizer, so we
        # need to precise for each one of them the specific variables to update.
        # Generator Network Variables

        with self.graph.as_default():
            batch_size_placeholder = tf.placeholder(tf.int64, name='batch_size')

            input_real = tf.placeholder(tf.float64, shape=(None, 60, 160, 3), name='input_real')
            input_fake = self.generator.tf_nodes['output']

            train_dataset = tf.data.Dataset.from_tensor_slices(tensors=(input_real))
            train_batch_dataset = train_dataset.batch(batch_size_placeholder).repeat()

            iterator = tf.data.Iterator.from_structure(train_batch_dataset.output_types,
                                                       train_batch_dataset.output_shapes)
            input_real_batch = iterator.get_next()

            learning_rate_placeholder = tf.placeholder(tf.float64, name='learning_rate')
            dropout_rate_placeholder = tf.placeholder(tf.float64, name='dropout_rate')

            output_real = self.discriminator.build_discriminator(input_real_batch, dropout_rate_placeholder,
                                                                 learning_rate_placeholder)
            output_fake = self.discriminator.build_discriminator(input_fake, dropout_rate_placeholder,
                                                                 learning_rate_placeholder)

            gen_loss = -tf.reduce_mean(tf.log(output_fake))
            disc_loss = -tf.reduce_mean(tf.log(output_real) + tf.log(1. - output_fake))

            gen_vars = [node for node in graph._collections['trainable_variables'] if node.name.startswith('gen')]
            disc_vars = [node for node in graph._collections['trainable_variables'] if node.name.startswith('disc')]

            global_step = tf.Variable(0, trainable=False)
            optimized_learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                 global_step=global_step,
                                                                 decay_steps=n_batch,
                                                                 decay_rate=decay_rate,
                                                                 staircase=True
                                                                 )
            optimizer_gen = tf.train.AdamOptimizer(learning_rate=optimized_learning_rate)
            optimizer_disc = tf.train.AdamOptimizer(learning_rate=optimized_learning_rate)

            train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
            train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars, global_step=global_step)

            init = tf.global_variables_initializer()
            self.train_iter_init_op = iterator.make_initializer(train_batch_dataset)

            self.sess.run(init)
            self.sess.run(self.train_iter_init_op,
                          feed_dict={
                              input_real: X,
                              batch_size_placeholder: batch_size})

            for i in range(epoch):
                for n in range(int(n_batch)):
                    generated_fake_noises = self.generator.generate_radom_noise(batch_size)
                    _, gen_cost, output_f = self.sess.run([train_gen, gen_loss, output_fake],
                                                          feed_dict={
                                                              self.generator.tf_nodes['x_placeholder']: generated_fake_noises,
                                                              learning_rate_placeholder: learning_rate,
                                                              dropout_rate_placeholder: dropout_rate
                                                          })
                    _, disc_cost, output_r = self.sess.run([train_disc, gen_loss, output_real],
                                                           feed_dict={
                                                               self.generator.tf_nodes['x_placeholder']: generated_fake_noises,
                                                               learning_rate_placeholder: learning_rate,
                                                               dropout_rate_placeholder: dropout_rate
                                                           })

                    if n % 10 == 0:
                        print(gen_cost, disc_cost, output_f[:5], output_r[:5])
                if i % 2 == 0:
                    sample = self.generator.generate_captcha(1)
                    plt.imshow(sample.reshape(60, 160, 3))
                    plt.show()


if __name__ == '__main__':
    from preprocess.load_data import get_data

    x, y = get_data(pardir='/home/fount/poc_server/notebooks/Captcha/data')
    print(x.shape, y.shape)

    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    gan = CaptchaGenAdvNet(graph=graph, sess=sess)
    gan.train(X=x[:10000], epoch=200, learning_rate=0.001, batch_size=100, dropout_rate=0.3)
    print(gan)
    import pdb;pdb.set_trace()
