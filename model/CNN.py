import tensorflow as tf


X = tf.placeholder(tf.float64, shape=(None, 60, 160, 3))
y = tf.placeholder(tf.float64, shape=(None, 5, 10))
conv1 = tf.layers.conv2d(inputs=X,
                        filters=32,
                        kernel_size=[5, 5],
                        padding='same',
                        activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
conv2 = tf.layers.conv2d(inputs=input_layer,
                        filters=64,
                        kernel_size=[5, 5],
                        padding='same',
                        activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
pool2_flat = tf.reshape(pool2, [-1, 30 * 80 * 64])
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=True)
logits = tf.layers.dense(inputs=dropout, units=50)

reshaped_logits = tf.reshape(logits, [-1, 5, 10])


if __name__ == '__main__':
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        output = sess.run(logits, feed_dict={X: input_data.reshape(-1, 60, 160, 3)/255})
    print(output)
