import tensorflow as tf
import numpy as np
from model.backbones import mobilenetv3_large
from model.backend_layers import upsampling_layer, conv2d
from Dataset import Dataset
import config as cfg

def FCN8s(input, num_classes, backbone_type, is_training, input_size=224):
    with tf.variable_scope('encoder') as scope:
        if backbone_type=='mobilenetv3_large':
            feature_8s, feature_16s, feature_32s = mobilenetv3_large(input=input, is_training=is_training, input_size=input_size)

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

        #return scores_32s, logits_16s, logits_8s
        return tf.identity(up_8s_to_ori, name='logits_output')


if __name__=="__main__":
    dataset = Dataset(cfg.TEST_SET)
    g1= tf.Graph()
    with g1.as_default():
        x = tf.placeholder(dtype=tf.float32, shape=[None,224,224,3])
        y = tf.placeholder(dtype=tf.int32)

        logits_output = FCN8s(input=x,num_classes=3,backbone_type='mobilenetv3_large',is_training=True)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits_output)

    with tf.Session(graph=g1) as sess:
        sess.run(tf.global_variables_initializer())
        for image, label, batch_count in dataset:
            l = sess.run(loss, feed_dict={x:image,
                                                         y:label})
            print(l.shape)


