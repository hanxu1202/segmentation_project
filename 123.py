import tensorflow as tf
import os
from tensorflow.python.client import device_lib
import numpy as np
with tf.device('/gpu:0'):
    x = tf.placeholder(dtype=tf.float32, shape=[1,224,224,3])
    y = tf.layers.conv2d(x, 64,5)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        v = sess.run(y, feed_dict={x:np.ones(shape=[1,224,224,3])})

