import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import tensorflow as tf
import cv2
import numpy as np
from config import PREDICT
from model.FCN import FCN
from utils.utils import draw_colored_mask, CRF_process


def predcit(image):
    g1 = tf.Graph()
    with g1.as_default():
        # define input
        x1 = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3])

        # forward
        if PREDICT.MODEL_TYPE == 'FCN':
            logits_output = FCN(x1, 3, backbone_type=PREDICT.BACKBONE_TYPE, is_training=False, input_size=224, ds_feature=PREDICT.DS_FEATURE)

        # predict
        softmax = tf.nn.softmax(logits_output, axis=-1, name='softmax')
        predcit = tf.squeeze(tf.argmax(softmax, axis=-1), name='predict')

        # loader
        load_var_list=[]
        for v in tf.global_variables():
            if 'FCN' in v.name:
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


if __name__=='__main__':    
    image = cv2.imread(PREDICT.IMAGE_DIR, -1)   # image.shape: h, w, c
    image_norm = (image/255 - PREDICT.NORM_MEAN) / PREDICT.NORM_STD
    img_resize = cv2.resize(image_norm, (PREDICT.IMG_SIZE, PREDICT.IMG_SIZE))
    img_resize = np.expand_dims(img_resize, 0)
    prob = predcit(img_resize)
    prob = np.squeeze(prob)
    prob = cv2.resize(prob, (image.shape[1], image.shape[0]))  # reseize=> w, h

    crf_mask = CRF_process(prob, image, PREDICT.NUM_CLASSES)

    color_map = {0: [0, 0, 0],
                 1: [0, 128, 128],
                 2: [128, 0, 128]}

    colored_img = draw_colored_mask(image, crf_mask, color_map)
    cv2.imshow('prediction', colored_img)
    cv2.waitKey(0)

