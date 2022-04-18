import tensorflow as tf
from model.backbones.mobilenetv3_16s import mobilenetv3_large_16s
from model.backbones.mobilenetv2_16s import mobilenetv2_16s
from utils.backend_layers import conv2d, upsampling_layer, v3ASPP_module


def Deeplabv3p(input, backbone_type, num_classes, is_training, input_size=224):
    with tf.variable_scope('Deeplabv3p') as scope:
        with tf.variable_scope('encoder') as scope:
            if backbone_type=='mobilenetv3_large_16s':
                feature_4s, feature_16s = mobilenetv3_large_16s(input=input, is_training=is_training)
            if backbone_type=='mobilenetv2_16s':
                feature_4s, feature_16s = mobilenetv2_16s(input=input, is_training=is_training)

        with tf.variable_scope('decoder') as scope:
            ASPP = v3ASPP_module(name='ASPP', x=feature_16s, input_size=input_size//16, input_channels=feature_16s.shape[-1],
                                 output_channels=256, nl='relu', rate=[6,12,18], padding='SAME', is_training=is_training)
            high_feature = upsampling_layer(name='16to4', x=ASPP, channels=0, input_size=input_size//16,
                                          upsampling_type='bilinear', upsample_rate=4)
            low_feature = conv2d(name='low_feature_conv2d', x=feature_4s, filter_shape=1, input_channels=feature_4s.shape[-1],
                                 output_channels=48, strides=(1,1), nl='relu', padding='SAME', use_bn=True, use_bias=False,
                                 activation=True, is_training=is_training)
            feature_concat = tf.concat([high_feature, low_feature], axis=-1)

            conv1 = conv2d(name='conv1', x=feature_concat, filter_shape=3, input_channels=feature_concat.shape[-1],
                           output_channels=256, strides=(1,1), nl='relu', padding='SAME', use_bn=True, use_bias=False,
                           activation=True, is_training=is_training)
            conv2 = conv2d(name='conv2', x=conv1, filter_shape=3, input_channels=256,
                           output_channels=256, strides=(1,1), nl='relu', padding='SAME', use_bn=True, use_bias=False,
                           activation=True, is_training=is_training)
            conv3 = conv2d(name='conv3', x=conv2, filter_shape=1, input_channels=256,
                           output_channels=num_classes, strides=(1,1), nl=None, padding='SAME', use_bn=False, use_bias=True,
                           activation=False, is_training=is_training)
            up_4s_to_ori = upsampling_layer(name='4toori', x=conv3, channels=0, input_size=input_size//4,
                                          upsampling_type='bilinear', upsample_rate=4)

            return tf.identity(up_4s_to_ori, name='logits_output')