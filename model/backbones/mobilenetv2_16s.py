import tensorflow as tf
import numpy as np
from utils.backend_layers import conv2d, depthwise_conv2d, v2Bottleneck
from tensorflow.python.framework import graph_util


def mobilenetv2_16s(input, is_training, input_size=224, num_classes=3):
    with tf.variable_scope('mobilenetv2_16s') as scope:
        head_conv1 = conv2d(name='head_conv1', x=input, filter_shape=3, input_channels=3, output_channels=32,
                            strides=2, nl='relu6', padding='SAME', use_bn=False, use_bias=True, activation=True,
                            is_training=is_training)  # /2
        head_dwconv = depthwise_conv2d(name='head_dwconv', x=head_conv1, channels=32, filter_shape=3, strides=(1, 1, 1, 1),
                                       nl='relu6', padding='SAME', use_bn=True, use_bias=False, activation=True,
                                       is_training=is_training)
        head_conv2 = conv2d(name='head_conv2', x=head_dwconv, filter_shape=1, input_channels=32, output_channels=16,
                            strides=(1, 1), padding='SAME', use_bn=False, nl=None, use_bias=True, activation=False,
                            is_training=is_training)
        bottleneck1_1 = v2Bottleneck(name='bottleneck1_1', x=head_conv2, input_channels=16, expand_channels=96,
                                     output_channels=24, dwconv_ksize=3, is_ds=True, is_training=is_training)  # /4
        bottleneck1_2 = v2Bottleneck(name='bottleneck1_2', x=bottleneck1_1, input_channels=24, expand_channels=144,
                                     output_channels=24, dwconv_ksize=3, is_ds=False, is_training=is_training)

        bottleneck2_1 = v2Bottleneck(name='bottleneck2_1', x=bottleneck1_2, input_channels=24, expand_channels=144,
                                     output_channels=32, dwconv_ksize=3, is_ds=True, is_training=is_training)  # /8
        bottleneck2_2 = v2Bottleneck(name='bottleneck2_2', x=bottleneck2_1, input_channels=32, expand_channels=192,
                                     output_channels=32, dwconv_ksize=3, is_ds=False, is_training=is_training)
        bottleneck2_3 = v2Bottleneck(name='bottleneck2_3', x=bottleneck2_2, input_channels=32, expand_channels=192,
                                     output_channels=32, dwconv_ksize=3, is_ds=False, is_training=is_training)

        bottleneck3_1 = v2Bottleneck(name='bottleneck3_1', x=bottleneck2_3, input_channels=32, expand_channels=192,
                                     output_channels=64, dwconv_ksize=3, is_ds=True, is_training=is_training)  # /16
        bottleneck3_2 = v2Bottleneck(name='bottleneck3_2', x=bottleneck3_1, input_channels=64, expand_channels=384,
                                     output_channels=64, dwconv_ksize=3, is_ds=False, is_training=is_training)
        bottleneck3_3 = v2Bottleneck(name='bottleneck3_3', x=bottleneck3_2, input_channels=64, expand_channels=384,
                                     output_channels=64, dwconv_ksize=3, is_ds=False, is_training=is_training)
        bottleneck3_4 = v2Bottleneck(name='bottleneck3_4', x=bottleneck3_3, input_channels=64, expand_channels=384,
                                     output_channels=64, dwconv_ksize=3, is_ds=False, is_training=is_training)

        bottleneck4_1 = v2Bottleneck(name='bottleneck4_1', x=bottleneck3_4, input_channels=64, expand_channels=384,
                                     output_channels=96, dwconv_ksize=3, is_ds=False, is_training=is_training)
        bottleneck4_2 = v2Bottleneck(name='bottleneck4_2', x=bottleneck4_1, input_channels=96, expand_channels=576,
                                     output_channels=96, dwconv_ksize=3, is_ds=False, is_training=is_training)
        bottleneck4_3 = v2Bottleneck(name='bottleneck4_3', x=bottleneck4_2, input_channels=96, expand_channels=576,
                                     output_channels=96, dwconv_ksize=3, is_ds=False, is_training=is_training)

        bottleneck5_1 = v2Bottleneck(name='bottleneck5_1', x=bottleneck4_3, input_channels=96, expand_channels=576,
                                     output_channels=160, dwconv_ksize=3, is_ds=False, rate=[2, 2],
                                     is_training=is_training)  # /16
        bottleneck5_2 = v2Bottleneck(name='bottleneck5_2', x=bottleneck5_1, input_channels=160, expand_channels=960,
                                     output_channels=160, dwconv_ksize=3, is_ds=False, is_training=is_training)
        bottleneck5_3 = v2Bottleneck(name='bottleneck5_3', x=bottleneck5_2, input_channels=160, expand_channels=960,
                                     output_channels=160, dwconv_ksize=3, is_ds=False, is_training=is_training)

        bottleneck6_1 = v2Bottleneck(name='bottleneck6_1', x=bottleneck5_3, input_channels=160, expand_channels=960,
                                     output_channels=320, dwconv_ksize=3, is_ds=False, is_training=is_training)

        return bottleneck1_2, bottleneck6_1



if __name__ == '__main__':
    g1 = tf.Graph()
    with g1.as_default():
        input = tf.placeholder(dtype=tf.float32, shape=[2, 224, 224, 3])
        x = mobilenetv2(input=input, is_training=False)
        for n in g1.as_graph_def().node:
            print(n.name)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['mobilenetv2_16s/bottleneck6_1/output'])
            with tf.gfile.FastGFile('../visualization/mobilenetv2_16s.pb', mode='wb') as f:
                f.write(constant_graph.SerializeToString())