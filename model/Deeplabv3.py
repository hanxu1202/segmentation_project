import tensorflow as tf
from model.backbones.mobilenetv3_16s import mobilenetv3_large_16s
from utils.backend_layers import conv2d, upsampling_layer, v3ASPP_module


def Deeplabv3(input, backbone_type, num_classes, is_training, input_size=224):
    with tf.variable_scope('Deeplabv3') as scope:
        with tf.variable_scope('encoder') as scope:
            if backbone_type=='mobilenetv3_large_16s':
                feature4s, feature_16s = mobilenetv3_large_16s(input=input, is_training=is_training)

        with tf.variable_scope('decoder') as scope:
            ASPP = v3ASPP_module(name='ASPP', x=feature_16s, input_size=input_size//16, input_channels=feature_16s.shape[-1],
                                 output_channels=256, nl='relu', rate=[6,12,18], padding='SAME', is_training=is_training)
            conv1 = conv2d(name='conv1', x=ASPP, filter_shape=1, input_channels=256,output_channels=num_classes,
                           strides=(1,1), nl=None, padding='SAME', use_bn=False, use_bias=True, activation=False,
                           is_training=is_training)
            up_16s_to_ori = upsampling_layer(name='16toori', x=conv1, channels=0, input_size=input_size//16,
                                          upsampling_type='bilinear', upsample_rate=16)

            return tf.identity(up_16s_to_ori, name='logits_output')