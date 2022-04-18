import tensorflow as tf
import numpy as np
from utils.backend_layers import conv2d, dense_layer,depthwise_conv2d, SEmodule, v3Bottleneck
from tensorflow.python.framework import graph_util


def mobilenetv3_large(input, is_training, input_size=224, num_classes=3):
    with tf.variable_scope('encoder') as scope:
        with tf.variable_scope('mobilenetv3_large') as scope:
            head_conv1 = conv2d(name='head_conv1', x=input, filter_shape=3, input_channels=3,output_channels=16,
                                     strides=2, nl='h_swish', padding='SAME', use_bn=True, use_bias=False, activation=True,
                                     is_training= is_training) # /2
            head_dwconv = depthwise_conv2d(name='head_dwconv', x=head_conv1, channels=16,filter_shape=3,strides=(1,1,1,1),nl='relu',
                                           padding='SAME', use_bn=True, use_bias=False, activation=True, is_training=is_training)
            head_conv2 = conv2d(name='head_conv2', x=head_dwconv, filter_shape=1, input_channels=16, output_channels=16,
                                strides=(1,1), padding='SAME', use_bn=True, nl=None, use_bias=False, activation=False,
                                is_training=is_training)

            short_cut = tf.identity(tf.add(head_conv1, head_conv2), name='head_shortcut')

            bottleneck1_1 = v3Bottleneck(name='bottleneck1_1', x=short_cut, input_channels=16, expand_channels=64,
                                         output_channels=24, dwconv_ksize=3, nl='relu', is_ds=True, use_se=False,
                                         is_training=is_training)  # /4
            bottleneck1_2 = v3Bottleneck(name='bottleneck1_2', x=bottleneck1_1, input_channels=24, expand_channels=72,
                                         output_channels=24, dwconv_ksize=3, nl='relu', is_ds=False, use_se=False,
                                         is_training=is_training)

            bottleneck2_1 = v3Bottleneck(name='bottleneck2_1', x=bottleneck1_2, input_channels=24, expand_channels=72,
                                         output_channels=40, dwconv_ksize=5, nl='relu', is_ds=True, use_se=True,
                                         mid_channels=24, pool_size=int(input_size//8), is_training=is_training)  # /8
            bottleneck2_2 = v3Bottleneck(name='bottleneck2_2', x=bottleneck2_1, input_channels=40, expand_channels=120,
                                         output_channels=40, dwconv_ksize=5, nl='relu', is_ds=False, use_se=True,
                                         mid_channels=32, pool_size=int(input_size//8), is_training=is_training)
            bottleneck2_3 = v3Bottleneck(name='bottleneck2_3', x=bottleneck2_2, input_channels=40, expand_channels=120,
                                         output_channels=40, dwconv_ksize=5, nl='relu', is_ds=False, use_se=True,
                                         mid_channels=32, pool_size=int(input_size//8), is_training=is_training)

            bottleneck3_1 = v3Bottleneck(name='bottleneck3_1', x=bottleneck2_3, input_channels=40, expand_channels=240,
                                         output_channels=80, dwconv_ksize=3, nl='h_swish', is_ds=True, use_se=False,
                                         is_training=is_training)  # /16
            bottleneck3_2 = v3Bottleneck(name='bottleneck3_2', x=bottleneck3_1, input_channels=80, expand_channels=200,
                                         output_channels=80, dwconv_ksize=3, nl='h_swish', is_ds=False, use_se=False,
                                         is_training=is_training)
            bottleneck3_3 = v3Bottleneck(name='bottleneck3_3', x=bottleneck3_2, input_channels=80, expand_channels=184,
                                         output_channels=80, dwconv_ksize=3, nl='h_swish', is_ds=False, use_se=False,
                                         is_training=is_training)
            bottleneck3_4 = v3Bottleneck(name='bottleneck3_4', x=bottleneck3_3, input_channels=80, expand_channels=184,
                                         output_channels=80, dwconv_ksize=3, nl='h_swish', is_ds=False, use_se=False,
                                         is_training=is_training)
            bottleneck3_5 = v3Bottleneck(name='bottleneck3_5', x=bottleneck3_4, input_channels=80, expand_channels=480,
                                         output_channels=112, dwconv_ksize=3, nl='h_swish', is_ds=False, use_se=True,
                                         mid_channels=120, pool_size=int(input_size // 16), is_training=is_training)
            bottleneck3_6 = v3Bottleneck(name='bottleneck3_6', x=bottleneck3_5, input_channels=112, expand_channels=672,
                                         output_channels=112, dwconv_ksize=3, nl='h_swish', is_ds=False, use_se=True,
                                         mid_channels=168, pool_size=int(input_size // 16), is_training=is_training)

            bottleneck4_1 = v3Bottleneck(name='bottleneck4_1', x=bottleneck3_6, input_channels=112, expand_channels=672,
                                         output_channels=160, dwconv_ksize=5, nl='h_swish', is_ds=True, use_se=True,
                                         mid_channels=168, pool_size=int(input_size // 32), is_training=is_training) #/32
            bottleneck4_2 = v3Bottleneck(name='bottleneck4_2', x=bottleneck4_1, input_channels=160, expand_channels=960,
                                         output_channels=160, dwconv_ksize=5, nl='h_swish', is_ds=False, use_se=True,
                                         mid_channels=240, pool_size=int(input_size // 32), is_training=is_training)
            bottleneck4_3 = v3Bottleneck(name='bottleneck4_3', x=bottleneck4_2, input_channels=160, expand_channels=960,
                                         output_channels=160, dwconv_ksize=5, nl='h_swish', is_ds=False, use_se=True,
                                         mid_channels=240, pool_size=int(input_size // 32), is_training=is_training)
    with tf.variable_scope('decoder'):
        conv1 = conv2d(name='conv1', x=bottleneck4_3, filter_shape=1, input_channels=160, output_channels=960, strides=1,
                       nl='h_swish', padding='SAME', use_bn=True,use_bias=False, activation=True, is_training=is_training)
        avg_pool = tf.nn.avg_pool2d(name='global_avgpool',value=conv1, ksize=input_size//32, strides=[1, 1], padding="VALID")
        conv2 = conv2d(name='conv2', x=avg_pool, filter_shape=1, input_channels=960, output_channels=1280,strides=1,
                       nl='h_swish', padding='SAME', use_bn=False, use_bias=True, activation=True, is_training=is_training)
        logits = conv2d(name='logits', x= conv2, filter_shape=1, input_channels=1280, output_channels=num_classes, strides=1,
                        nl=None, padding='SAME', use_bn=False, use_bias=True, activation=False, is_training=is_training)

        return logits