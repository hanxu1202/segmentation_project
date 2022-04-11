import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import tensorflow as tf
import cv2
import numpy as np
from config import PREDICT
from model.FCN8s import FCN8s
from utils.utils import draw_colored_mask

def predcit(image):
    g1 = tf.Graph()
    with g1.as_default():
        # define input
        x1 = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3])

        # forward
        if PREDICT.MODEL_TYPE == 'FCN8s':
            logits_output = FCN8s(x1, 3, backbone_type=PREDICT.BACKBONE_TYPE, is_training=False, input_size=224)

        # predict
        softmax = tf.nn.softmax(logits_output, axis=-1, name='softmax')
        predcit = tf.squeeze(tf.arg_max(softmax, axis=-1), name='predict')

        # loader
        load_var_list=[]
        for v in tf.global_variables():
            if 'FCN8s' in v.name:
                load_var_list.append(v)

        loader = tf.train.Saver(var_list=load_var_list)

        # initializer
        init = tf.global_variables_initializer()

        # session config
        sess_cfg = tf.ConfigProto(allow_soft_placement=True)
        sess_cfg.gpu_options.allow_growth = True

        with tf.Session(config=sess_cfg, graph=g1) as sess:
            sess.run(init)
            pred = sess.run(predcit, feed_dict={x1: image})

        return pred


if __name__=='__main__':    
    image = cv2.imread(PREDICT.IMAGE_DIR, -1)
    w, h = image.shape[0:2]
    image_norm = (image/255 - PREDICT.NORM_MEAN) / PREDICT.NORM_STD
    img_resize = cv2.resize(image_norm, (PREDICT.IMG_SIZE, PREDICT.IMG_SIZE))
    pred = predcit(img_resize)

    ori_mask = cv2.resize(pred, (w, h))

    color_map = {0: [0, 0, 0],
                 1: [0, 128, 128],
                 2: [128, 0, 128]}

    colored_img = draw_colored_mask(image, ori_mask, color_map)
    cv2.imshow('dad', colored_img)
    cv2.waitKey(0)