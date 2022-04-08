import tensorflow as tf
from tensorflow.python.framework import graph_util


def get_weights(shape, initializer=tf.glorot_normal_initializer(), weight_decay=0.001, trainable=True):
    filters = tf.get_variable(name='weights',
                              shape=shape,
                              dtype=tf.float32,
                              initializer=initializer,
                              trainable=trainable)
    if weight_decay != 0:
        reg_loss = tf.contrib.layers.l2_regularizer(weight_decay)(filters)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, reg_loss)

    return filters


def get_bias(shape, initializer=tf.constant_initializer(0.0), trainable=True):
    bias = tf.get_variable(name='bias',
                           shape=shape,
                           dtype=tf.float32,
                           initializer=initializer,
                           trainable=trainable)
    return bias


def h_swish(x,
            name="h_swish"):
    with tf.variable_scope(name):
        return tf.identity(tf.multiply(tf.multiply(tf.nn.relu6(tf.add(x,3)), x), 1/6),name='output')


def h_sigmoid(x,
              name="h_sigmoid"):
    with tf.variable_scope(name):
        return tf.identity(tf.multiply(tf.nn.relu6(tf.add(x,3)), 1/6), name='output')

def relu(x,
         name="relu"):
    with tf.variable_scope(name):
       return tf.nn.relu(x)


def relu6(x,
          name="relu6"):
    with tf.variable_scope(name):
        return tf.identity(tf.nn.relu6(x), name='output')


def nl_layer(x,
               nl):
    assert nl in ['relu','relu6','h_swish','h_sigmoid']
    if nl == 'relu':
        return relu(x, name=nl)
    if nl == 'relu6':
        return relu6(x, name=nl)
    if nl == 'h_swish':
        return h_swish(x, name=nl)
    if nl =='h_sigmoid':
        return h_sigmoid(x, name=nl)


def conv2d(name,
           x,
           filter_shape,
           input_channels,
           output_channels,
           strides,
           nl,
           padding='SAME',
           use_bn=True,
           use_bias=False,
           activation=True,
           is_training=True):
    assert nl in ['relu','relu6','h_swish','h_sigmoid', None]
    with tf.variable_scope(name) as scope:
        filter = get_weights(shape=[filter_shape,filter_shape,input_channels,output_channels])
        conv = tf.nn.conv2d(x, filter=filter, strides=strides, padding=padding, name='conv')
        if use_bn:
            bn = tf.layers.batch_normalization(conv, training=is_training, name='bn')
            if activation:
                return tf.identity(nl_layer(bn, nl), name='output')
            else:
                return tf.identity(bn, name='output')
        elif use_bias:
            bias = get_bias(shape=output_channels)
            bias_add = tf.nn.bias_add(conv, bias, name='bias_add')
            if activation:
                return tf.identity(nl_layer(bias_add, nl), name='output')
            else:
                return tf.identity(bias_add, name='output')
        elif activation:
            return tf.identity(nl_layer(conv, nl), name='output')
        else:
            return tf.identity(conv, name='output')


def depthwise_conv2d(name,
                     x,
                     channels,
                     filter_shape,
                     strides,
                     nl,
                     padding='SAME',
                     use_bn=True,
                     use_bias=False,
                     activation=True,
                     is_training=True):
    assert nl in ['relu', 'relu6', 'h_swish', 'h_sigmoid', None]
    with tf.variable_scope(name) as scope:
        filter = get_weights(shape=[filter_shape,filter_shape,channels, 1])
        dpconv = tf.nn.depthwise_conv2d(input=x, filter=filter, strides=strides, padding=padding,name='dwconv2d')
        if use_bn:
            bn = tf.layers.batch_normalization(inputs=dpconv, training=is_training, name='bn')
            if activation:
                return tf.identity(nl_layer(bn, nl), name='output')
            else:
                return tf.identity(bn, name='output')
        elif use_bias:
            bias = get_bias(shape=channels)
            bias_add = tf.nn.bias_add(dpconv, bias, name='bias_add')
            if activation:
                return tf.identity(nl_layer(bias_add, nl), name='output')
            else:
                return tf.identity(bias_add, name='output')
        elif activation:
            return tf.identity(nl_layer(dpconv, nl), name='output')
        else:
            return tf.identity(dpconv, name='output')

def dense_layer(name,
                x,
                input_channels,
                output_channels,
                nl,
                padding='SAME',
                use_bn=False,
                use_bias=True,
                activation=True,
                is_training=True,
                ):
    with tf.variable_scope(name) as scope:
        conv = conv2d('conv', x, filter_shape=1, strides=(1,1), padding=padding,
                      input_channels=input_channels, output_channels=output_channels,
                      nl=nl, use_bias=use_bias, use_bn=use_bn,
                      activation=activation, is_training=is_training)

        return tf.identity(conv, name='output')


def upsampling_layer(name,
                     x,
                     channels,
                     input_size,
                     upsampling_type,
                     filter_shape=None,
                     upsample_rate=2,
                     strides=(2, 2),
                     padding='SAME',
                     use_bn=True,
                     use_bias=False,
                     nl='relu',
                     activation=True,
                     is_training=True
                     ):
    with tf.variable_scope(name):
        if upsampling_type=='conv2d_transpose':
            filter = get_weights(shape=[filter_shape,filter_shape,channels,channels])
            conv_transpose = tf.nn.conv2d_transpose(name='conv_transpose', value=x, filter=filter,
                                                    output_shape=input_size * upsample_rate, strides=strides,
                                                    padding= padding)
            if use_bn:
                bn = tf.layers.batch_normalization(inputs=conv_transpose, training=is_training, name='bn')
                if activation:
                    return tf.identity(nl_layer(bn, nl), name='output')
                else:
                    return tf.identity(bn, name='output')
            elif use_bias:
                bias = get_bias(shape=channels)
                bias_add = tf.nn.bias_add(conv_transpose, bias, name='bias_add')
                if activation:
                    return tf.identity(nl_layer(bias_add, nl), name='output')
                else:
                    return tf.identity(bias_add, name='output')
            elif activation:
                return tf.identity(nl_layer(conv_transpose, nl), name='output')
            else:
                return tf.identity(conv_transpose, name='output')
        if upsampling_type=='bilinear':
            bilinear = tf.image.resize_bilinear(name='bilinear_layer', images=x,
                                                size=(input_size*upsample_rate,input_size*upsample_rate))
            return tf.identity(bilinear, name='output')

def SEmodule(x,
             name,
             pool_size,
             in_channels,
             mid_channels,
             is_training=True,
             ):
    with tf.variable_scope(name):
        avg_pool = tf.nn.avg_pool2d(value=x, ksize=pool_size, strides=[1, 1], padding="VALID",
                             name="global_avgpool")

        dense1 = dense_layer(name='dense1', x=avg_pool, input_channels=in_channels, output_channels=mid_channels, nl='relu',
                        padding='SAME', use_bn=False, use_bias=True, activation=True, is_training=is_training)
        dense2 = dense_layer(name='dense2', x=dense1, input_channels=mid_channels, output_channels=in_channels, nl='h_sigmoid',
                        padding='SAME', use_bn=False, use_bias=True, activation=True, is_training=is_training)

        mul = tf.multiply(dense2, x, name='multiply')
        return tf.identity(mul, name='SE_output')


def v3Bottleneck(name,
                 x,
                 input_channels,
                 expand_channels,
                 output_channels,
                 dpconv_ksize,
                 nl,
                 is_ds,
                 use_se=False,
                 mid_channels=None,
                 pool_size=None,
                 padding='SAME',
                 is_training=True):
    with tf.variable_scope(name):
        expand_conv = conv2d(name='expand_conv',x=x,filter_shape=1, input_channels=input_channels,
                            output_channels=expand_channels, strides=(1,1), nl=nl, use_bn=True, use_bias=False,
                            activation=True, is_training=is_training)
        if is_ds:
            dp_conv = depthwise_conv2d(name='dp_conv', x=expand_conv, channels=expand_channels,
                                       filter_shape=dpconv_ksize, strides=(1,2,2,1), nl=nl, padding=padding,
                                       use_bn=True, use_bias=False, is_training=is_training)
        else:
            dp_conv = depthwise_conv2d(name='dp_conv', x=expand_conv, channels=expand_channels,
                                       filter_shape=dpconv_ksize, strides=(1,1,1,1), nl=nl, padding=padding,
                                       use_bn=True, use_bias=False, is_training=is_training)
        if use_se:
            se = SEmodule(name='SEmodule',x=dp_conv, pool_size=(pool_size,pool_size),in_channels=expand_channels,
                          mid_channels=mid_channels, is_training=is_training)
        else:
            se = tf.identity(dp_conv, name='noSE')

        proj_conv = conv2d(name='proj_conv', x=se, filter_shape=1, input_channels=expand_channels,
                           output_channels=output_channels, strides=(1,1), nl=nl, padding=padding,
                           use_bn=True, use_bias=False, activation=False, is_training=is_training)

        if input_channels == output_channels and not is_ds:
            short_cut = tf.add(proj_conv, x, name="short_cut")
            return tf.identity(short_cut, name='output')
        else:
            return tf.identity(proj_conv, name='output')


if __name__=='__main__':
    g1 = tf.Graph()
    with g1.as_default():
        input = tf.get_variable(name='input',shape=[1,112,112,24])
        x = v3Bottleneck(name='Bottlenet1', x=input, input_channels=24, expand_channels=72,output_channels=24,
                         dpconv_ksize=3, nl='relu', is_ds=False, is_training=True)
        l2 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        for n in g1.as_graph_def().node:
            print(n.name)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            l = sess.run(l2)
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['Bottlenet1/output'])
            with tf.gfile.FastGFile('./model.pb', mode='wb') as f:
                f.write(constant_graph.SerializeToString())