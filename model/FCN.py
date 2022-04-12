import tensorflow as tf
import numpy as np
from model.backbones import mobilenetv3_large
from model.backend_layers import upsampling_layer, conv2d
from dataset import Dataset
import config as cfg

def FCN(input, num_classes, backbone_type, is_training, input_size=224, ds_feature=8):
    with tf.variable_scope('FCN') as scope:
        with tf.variable_scope('encoder') as scope:
            if backbone_type=='mobilenetv3_large':
                feature_4s, feature_8s, feature_16s, feature_32s = mobilenetv3_large(input=input, is_training=is_training, input_size=input_size)

        with tf.variable_scope('decoder') as scope:
            scores_32s = conv2d(name='scores_32s', x=feature_32s, filter_shape=1, input_channels=160,
                                output_channels=num_classes, strides=(1, 1), nl=None, padding='SAME', use_bn=False,
                                use_bias=False, activation=False, is_training=True)
            up_32s_to_16s = upsampling_layer(name='32to16', x=scores_32s, channels=num_classes, input_size=input_size//32,
                                             upsampling_type='bilinear', upsample_rate=2)
            scores_16s = conv2d(name='scores_16s', x=feature_16s, filter_shape=1, input_channels=112,
                                output_channels=num_classes, strides=(1, 1), nl=None, padding='SAME', use_bn=False,
                                use_bias=False, activation=False, is_training=True)
            logits_16s = tf.add(name='logits_16s', x=up_32s_to_16s, y=scores_16s)
            up_16s_to_8s = upsampling_layer(name='16to8', x=logits_16s, channels=num_classes, input_size=input_size//16,
                                            upsampling_type='bilinear', upsample_rate=2)
            scores_8s = conv2d(name='scores_8s', x=feature_8s, filter_shape=1, input_channels=40,
                                output_channels=num_classes, strides=(1, 1), nl=None, padding='SAME', use_bn=False,
                                use_bias=False, activation=False, is_training=True)
            logits_8s = tf.add(name='logits_8s', x=up_16s_to_8s, y=scores_8s)
            up_8s_to_ori = upsampling_layer(name='8toori', x=logits_8s, channels=num_classes, input_size=input_size//8,
                                      upsampling_type='bilinear', upsample_rate=8)

            up_8s_to_4s = upsampling_layer(name='8to4', x=logits_8s, channels=num_classes, input_size=input_size//8,
                                            upsampling_type='bilinear', upsample_rate=2)
            scores_4s = conv2d(name='scores_4s', x=feature_4s, filter_shape=1, input_channels=24,
                                output_channels=num_classes, strides=(1, 1), nl=None, padding='SAME', use_bn=False,
                                use_bias=False, activation=False, is_training=True)
            logits_4s = tf.add(name='logits_4s', x=up_8s_to_4s, y=scores_4s)

            up_4s_to_ori = upsampling_layer(name='4toori', x=logits_4s, channels=num_classes, input_size=input_size//4,
                                      upsampling_type='bilinear', upsample_rate=4)

            if ds_feature == 8:
                return tf.identity(up_8s_to_ori, name='logits_output')
            if ds_feature == 4:
                return tf.identity(up_4s_to_ori, name='logits_output')


if __name__=="__main__":
    dataset = Dataset(cfg.TEST_SET)
    g1= tf.Graph()
    with g1.as_default():
        x = tf.placeholder(dtype=tf.float32, shape=[None,224,224,3])
        y = tf.placeholder(dtype=tf.int32)

        logits_output = FCN(input=x,num_classes=3,backbone_type='mobilenetv3_large',is_training=True)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits_output)

    with tf.Session(graph=g1) as sess:
        sess.run(tf.global_variables_initializer())
        for image, label, batch_count in dataset:
            l = sess.run(loss, feed_dict={x:image,
                                                         y:label})
            print(l.shape)


