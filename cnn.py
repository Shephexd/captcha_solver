import tensorflow as tp
import numpy as np

#(160, 60, 3) -> after padding 1 (162,62, 3)
#fiter size = 6 filter 4*4*3
#stride = 2
#(N - F) / stride + 1 => (78, 27)
#padding + 1 


w = tf.Variable(tf.random_normal([4,4,3,60],stddev=0.01) #filters(4*4*3)
l1a = tf.nn.relu(tf.nn.conv2d(X, w, [1, 1, 1, 1], 'SAME'))
l1 = tf.nn.max_pool(c1, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1],pading='SAME')
