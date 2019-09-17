import scipy.misc
import numpy as np
import tensorflow as tf


X = tf.placeholder(tf.float64, shape=(None, 60, 160, 3))
y = tf.placeholder(tf.float64, shape=(None, 5, 10))

conv1 = tf.layers.conv2d(inputs=X,
                        filters=32,
                        kernel_size=[5, 5],
                        padding='same',
                        activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
conv2 = tf.layers.conv2d(inputs=pool1,
                        filters=64,
                        kernel_size=[5, 5],
                        padding='same',
                        activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

pool2_flat = tf.reshape(pool2, [-1, 15 * 40 * 64])
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=True)
logits = tf.layers.dense(inputs=dropout, units=50)
reshaped_logits = tf.reshape(logits, [-1, 5, 10])

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=reshaped_logits, labels=y)
loss = tf.reduce_sum(cross_entropy)

learning_rate=0.001
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_op = optimizer.minimize(loss=loss)


if __name__ == '__main__':
    # X:(N, 60, 160, 3)
    # Y:(N, 5, 10)
    input_data = scipy.misc.imread('42119.png')
    label = np.eye(10)[np.array([int(i) for i in '42119'])]
    labels = label.reshape(-1, 5, 10)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
    
        for i in range(10):
            print(sess.run([train_op, loss], feed_dict={X: input_data.reshape(-1, 60, 160, 3)/255, y: labels}))
        temp = sess.run(reshaped_logits, feed_dict={X: input_data.reshape(-1, 60, 160, 3)/255})
        print(temp.argmax(axis=2))

