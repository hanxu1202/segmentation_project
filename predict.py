import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import tensorflow as tf
import cv2
import numpy as np
from config import PREDICT
from model.FCN import FCN
from model.Deeplabv3p import Deeplabv3p
from model.Deeplabv3 import Deeplabv3
from utils.utils import draw_colored_mask, CRF_process
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util


def predcit(image):
    g1 = tf.Graph()
    with g1.as_default():
        # define input
        x1 = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3])

        # forward
        if PREDICT.MODEL_TYPE == 'FCN':
            logits_output = FCN(x1, num_classes=3, backbone_type=PREDICT.BACKBONE_TYPE, is_training=False, input_size=224, ds_feature=PREDICT.DS_FEATURE)
        if PREDICT.MODEL_TYPE == 'Deeplabv3':
            logits_output = Deeplabv3(x1, num_classes=3, backbone_type=PREDICT.BACKBONE_TYPE, is_training=False, input_size=224)
        if PREDICT.MODEL_TYPE == 'Deeplabv3p':
            logits_output = Deeplabv3p(x1, num_classes=3, backbone_type=PREDICT.BACKBONE_TYPE, is_training=False, input_size=224)

        # predict
        softmax = tf.nn.softmax(logits_output, axis=-1, name='softmax')
        predcit = tf.squeeze(tf.argmax(softmax, axis=-1), name='predict')

        # loader
        load_var_list=[]
        for v in tf.global_variables():
            if PREDICT.MODEL_TYPE in v.name:
                load_var_list.append(v)

        loader = tf.train.Saver(var_list=load_var_list)

        # initializer
        init = tf.global_variables_initializer()

        # session config
        sess_cfg = tf.ConfigProto(allow_soft_placement=True)
        sess_cfg.gpu_options.allow_growth = True

        with tf.Session(config=sess_cfg, graph=g1) as sess:
            sess.run(init)
            loader.restore(sess, PREDICT.MODEL_CKPT)
            sm, pred = sess.run([softmax, predcit], feed_dict={x1: image})

        return sm

def predcit_with_pb(image, pb_dir, x1_tensor, x2_tensor, ouput_tensor):
    g1 = tf.Graph()
    with g1.as_default():
        with tf.Session(graph=g1) as sess:
            with gfile.FastGFile(pb_dir, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()
                tf.import_graph_def(graph_def, name='')

            sess.run(tf.global_variables_initializer())

            x1 = g1.get_tensor_by_name(x1_tensor)
            x2 = g1.get_tensor_by_name(x2_tensor)
            output = g1.get_tensor_by_name(ouput_tensor)
            softmax = tf.nn.softmax(output)
            sm = sess.run(softmax, feed_dict={x1: image,
                                              x2: False})
            return sm


if __name__=='__main__':
    image = cv2.imread(PREDICT.IMAGE_DIR, -1)   # image.shape: h, w, c
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB
    image_norm = (image/255 - PREDICT.NORM_MEAN) / PREDICT.NORM_STD
    img_resize = cv2.resize(image_norm, (PREDICT.IMG_SIZE, PREDICT.IMG_SIZE))
    img_resize = np.expand_dims(img_resize, 0)
    prob = predcit(img_resize)
    '''
    pb_dir = 'E:\pycharm_project\segmentation_project\checkpoints\Deeplabv3p\\Deeplabv3p_mobilenetv2_16s_3c.pb'
    x1_tensor = 'input_placeholders/image_data:0'
    x2_tensor = 'input_placeholders/is_training:0'
    output_tensor = 'Deeplabv3p/decoder/logits_output:0'
    prob = predcit_with_pb(img_resize, pb_dir, x1_tensor, x2_tensor, output_tensor)
    '''

    prob = np.squeeze(prob)
    prob = cv2.resize(prob, (image.shape[1], image.shape[0]))  # reseize=> w, h

    ori_mask = np.squeeze(np.argmax(prob, axis=-1))
    crf_mask = CRF_process(prob, image, PREDICT.NUM_CLASSES, 5)

    color_map = {0: [0, 0, 0],
                 1: [0, 128, 128],
                 2: [128, 0, 128]}

    colored_img = draw_colored_mask(image, crf_mask, color_map)
    colored_img = cv2.cvtColor(colored_img, cv2.COLOR_RGB2BGR)  # BGR
    cv2.imshow('prediction', colored_img)
    cv2.waitKey(0)

