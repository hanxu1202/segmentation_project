import tensorflow as tf
import os
from tensorflow.python.client import device_lib
import numpy as np
from Dataset import Dataset
import config as cfg

testset = Dataset(cfg.TEST_SET)
for image, label, batchcount in testset:
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=image)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    l = sess.run(loss)