import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import tensorflow as tf
import numpy as np
from model.FCN8s import FCN8s
from dataset import Dataset
from config import TRAIN, TRAIN_SET, VAL_SET
from utils.utils import compute_miou
import logging



LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

logging.basicConfig(filename='./train_FCN8s_mobilenetv3_large.log', filemode='w', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)


def train(trainset, valset):
    with tf.device('/gpu:0'):
        g1 = tf.Graph()
        with g1.as_default():
            # define input
            with tf.name_scope("input_placeholders"):
                x1 = tf.placeholder(dtype=tf.float32,shape=[None, TRAIN_SET.IMG_SIZE, TRAIN_SET.IMG_SIZE, 3], name="image_data")
                x2 = tf.placeholder(dtype=tf.int32,name="label_data")
                x3 = tf.placeholder(dtype=tf.bool, name="is_training")
                x4 = tf.placeholder(dtype=tf.int32, name='epoch_number')
                x5 = tf.placeholder(dtype=tf.int32, name='global_step')

            # forward
            if TRAIN.MODEL_TYPE == "FCN8s":
                logits = FCN8s(input=x1, num_classes=TRAIN.NUM_CLASSES, backbone_type=TRAIN.BACKBONE_TYPE, is_training=x3, input_size=224)
                squeeze = tf.squeeze(logits)

            # define loss
            with tf.name_scope("loss"):
                batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=x2, logits=squeeze, name="batch_loss")
                mean_loss = tf.reduce_mean(batch_loss, name="mean_loss")

            # define predict
            with tf.name_scope("predict"):
                softmax = tf.nn.softmax(logits=logits, axis=-1, name="softmax")
                predict = tf.squeeze(tf.argmax(softmax, axis=-1, name='predict'))

            # define learning_rate
            with tf.name_scope("global_step_and_learning_rate"):
                learning_rate = tf.train.exponential_decay(TRAIN.INIT_LR, x5, TRAIN.LR_DECAY_STEP, TRAIN.LR_DECAY_RATE, staircase=True, name="learning_rate")

            # define optimizer
            with tf.name_scope('optimizer'):
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            # compute_gradients
            with tf.name_scope('compute_gradients'):
                grads_and_vars= optimizer.compute_gradients(mean_loss,var_list=tf.trainable_variables())

            # define train_op:
            with tf.name_scope("train_op"):
                train_op = optimizer.apply_gradients(grads_and_vars)
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    with tf.control_dependencies([train_op]):
                            train_iter = tf.no_op()

            # define saver and loader
            with tf.name_scope("saver_loader"):
                save_var = tf.global_variables()
                model_var = tf.global_variables()
                backbone_var=[]
                for var in tf.global_variables():
                    if 'encoder' in var.name:
                        backbone_var.append(var)
                saver = tf.train.Saver(var_list=save_var, max_to_keep=TRAIN.TOTAL_EPOCH)
                model_loader = tf.train.Saver(var_list=model_var)
                backbone_loader = tf.train.Saver(var_list=backbone_var)


            # define initializer
            with tf.name_scope("initializer"):
                init = tf.global_variables_initializer()

        # session config
        sess_cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        sess_cfg.gpu_options.allow_growth = True

        # start_train
        with tf.Session(graph=g1, config=sess_cfg) as sess:
            sess.run(init)
            if TRAIN.RESUME_CKPT is not None:
                try:
                    logging.info('=> Restoring model weights from: %s ... ' % TRAIN.RESUME_CKPT)
                    model_loader.restore(sess, TRAIN.RESUME_CKPT)
                except:
                    logging.info('=> %s does not exist !!!' % TRAIN.RESUME_CKPT)
                    logging.info('=> Now it starts to train from scratch ...')
            elif TRAIN.RESUME_CKPT is None and TRAIN.PRETRAINED_BACKBONE is not None:
                try:
                    logging.info('=> Restoring backbone weights from: %s ... ' % TRAIN.PRETRAINED_BACKBONE)
                    backbone_loader.restore(sess, TRAIN.PRETRAINED_BACKBONE)
                except:
                    logging.info('=> %s does not exist !!!' % TRAIN.RESUME_CKPT)
                    logging.info('=> Now it starts to train from scratch ...')
            else:
                logging.info('=> Now it starts to train from scratch ...')

            # list all variables in graph.
            logging.info('========================================================================================')
            logging.info('LIST ALL TRAINABLE VARIABLES')
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            for v in variables:
                logging.info('name: ' + str(v.name) + '  shape: ' + str(v.shape))
            logging.info('========================================================================================')

            total_step = TRAIN.TOTAL_EPOCH * trainset.batch_num
            for epoch in range(TRAIN.START_EPOCH, TRAIN.TOTAL_EPOCH+1):
                train_epoch_loss = []
                train_epoch_miou = []
                for image, label, batch_count in trainset:
                    global_step = (epoch-1) * trainset.batch_num + batch_count
                    train_pre, train_batch_loss, lr, _ = sess.run([predict, mean_loss, learning_rate, train_iter],
                                                            feed_dict={x1: image,
                                                                       x2: label,
                                                                       x3: True,
                                                                       x4: epoch,
                                                                       x5: global_step,})

                    train_epoch_loss.append(train_batch_loss)
                    train_batch_miou = compute_miou(train_pre, label, TRAIN.NUM_CLASSES)
                    train_epoch_miou.append(train_batch_miou)
                    if batch_count % TRAIN.SUMMARY_STEP == 0:
                        logging.info("global step: %d, epoch_%d, lr: %.5f, %d/%d: Train average loss: %.3f" %(global_step, epoch, lr, batch_count, trainset.batch_num, np.mean(train_epoch_loss)))

                logging.info('==========================================================')
                train_miou = np.mean(train_epoch_miou)
                logging.info("epoch_%d: Train set(Aug) miou is: %.3f" % (epoch, train_miou))
                if epoch % TRAIN.SAVE_EPOCH == 0:
                    saver.save(sess, TRAIN.SAVE_DIR + "FCN8s_"+TRAIN.BACKBONE_TYPE + ".ckpt", global_step=global_step)

                if epoch % TRAIN.VALID_EPOCH == 0:
                    valid_epoch_miou = []
                    valid_epoch_loss = []
                    for image, label, batch_count in valset:
                        val_pre, valid_batch_loss = sess.run([predict, mean_loss],feed_dict={x1: image,
                                                                                         x2: label,
                                                                                         x3: False,
                                                                                         x4: 0,
                                                                                         x5: 0,})
                        val_batch_miou = compute_miou(val_pre, label, TRAIN.NUM_CLASSES)
                        valid_epoch_miou.append(val_batch_miou)
                        valid_epoch_loss.append(valid_batch_loss)
                    valid_miou = np.mean(valid_epoch_miou)
                    valid_mean_loss = np.mean(valid_epoch_loss)
                    logging.info("epoch_%d: Valid set average loss is: %.3f" % (epoch, valid_mean_loss))
                    logging.info("epoch_%d: Valid set miou is: %.3f" % (epoch, valid_miou))
                    logging.info('==========================================================')





if __name__=='__main__':
    trianset = Dataset(TRAIN_SET)
    validset = Dataset(VAL_SET)
    train(trianset,validset)