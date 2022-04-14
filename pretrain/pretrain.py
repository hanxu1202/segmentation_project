import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import tensorflow as tf
import numpy as np
from pretrain_model import mobilenetv3_large, mobilenetv3_large_16s
from pretrain_dataset import Dataset
from pretrain_config import PRETRAIN, PRETRAIN_SET, PREVAL_SET
import logging
import datetime


def train(trainset, valset):
    # logfile settings.
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

    filename = './' + str(datetime.date.today()) + '_pretrain_' + PRETRAIN.MODEL_TYPE + '.log'
    logging.basicConfig(filename=filename, filemode='w', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    logging.info('Train configuration:')
    for key, value in PRETRAIN.items():
        logging.info(key + ': ' + str(value))

    with tf.device('/gpu:0'):
        g1 = tf.Graph()
        with g1.as_default():
            # define input
            with tf.name_scope("input_placeholders"):
                x1 = tf.placeholder(dtype=tf.float32,shape=[None, PRETRAIN_SET.IMG_SIZE, PRETRAIN_SET.IMG_SIZE, 3], name="image_data")
                x2 = tf.placeholder(dtype=tf.int32,name="label_data")
                x3 = tf.placeholder(dtype=tf.bool, name="is_training")
                x4 = tf.placeholder(dtype=tf.int32, name='epoch_number')
                x5 = tf.placeholder(dtype=tf.int32, name='global_step')

            # forward
            if PRETRAIN.MODEL_TYPE == "mobilenetv3_large":
                logits = mobilenetv3_large(input=x1, is_training=x3, input_size=PRETRAIN_SET.IMG_SIZE, num_classes=PRETRAIN.NUM_CLASSES)
                squeeze = tf.squeeze(logits)
            if PRETRAIN.MODEL_TYPE == "mobilenetv3_large_16s":
                logits = mobilenetv3_large_16s(input=x1, is_training=x3, input_size=PRETRAIN_SET.IMG_SIZE, num_classes=PRETRAIN.NUM_CLASSES)
                squeeze = tf.squeeze(logits)

            # define loss
            with tf.name_scope("loss"):
                batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=x2, logits=squeeze, name="batch_loss")
                mean_loss = tf.reduce_mean(batch_loss, name="mean_loss")

            # define predict
            with tf.name_scope("predict"):
                softmax = tf.nn.softmax(logits=logits, axis=-1, name="softmax")
                predict = tf.argmax(softmax, axis=-1, name='predict')

            # define learning_rate
            with tf.name_scope("global_step_and_learning_rate"):
                learning_rate = tf.train.exponential_decay(PRETRAIN.INIT_LR, x5, PRETRAIN.LR_DECAY_STEP, PRETRAIN.LR_DECAY_RATE, staircase=True, name="learning_rate")

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
                bn_moving_vars = [var for var in tf.global_variables() if 'moving_mean' in var.name]
                bn_moving_vars += [var for var in tf.global_variables() if 'moving_variance' in var.name]
                load_var = tf.trainable_variables() + bn_moving_vars
                saver = tf.train.Saver(var_list=save_var, max_to_keep=PRETRAIN.TOTAL_EPOCH)
                loader = tf.train.Saver(var_list=load_var)


            # define initializer
            with tf.name_scope("initializer"):
                init = tf.global_variables_initializer()

        # session config
        sess_cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        sess_cfg.gpu_options.allow_growth = True

        # start_train
        with tf.Session(graph=g1, config=sess_cfg) as sess:
            sess.run(init)
            try:
                logging.info('=> Restoring weights from: %s ... ' % PRETRAIN.RESUME_CKPT)
                loader.restore(sess, PRETRAIN.RESUME_CKPT)
            except:
                logging.info('=> %s does not exist !!!' % PRETRAIN.RESUME_CKPT)
                logging.info('=> Now it starts to train from scratch ...')

            # list all variables in graph.
            logging.info('========================================================================================')
            logging.info('LIST ALL TRAINABLE VARIABLES')
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            for v in variables:
                logging.info('name: ' + str(v.name) + '  shape: ' + str(v.shape))
            logging.info('========================================================================================')

            total_step = PRETRAIN.TOTAL_EPOCH * trainset.batch_num
            for epoch in range(PRETRAIN.START_EPOCH, PRETRAIN.TOTAL_EPOCH+1):
                train_epoch_loss = []
                train_epoch_acc = []
                for image, label, batch_count in trainset:
                    global_step = (epoch-1) * trainset.batch_num + batch_count
                    train_pre, train_batch_loss, lr, _ = sess.run([predict, mean_loss, learning_rate, train_iter],
                                                            feed_dict={x1: image,
                                                                       x2: label,
                                                                       x3: True,
                                                                       x4: epoch,
                                                                       x5: global_step,})

                    train_epoch_loss.append(train_batch_loss)
                    train_batch_acc = np.mean(np.where(np.squeeze(train_pre) == label, 1, 0))
                    train_epoch_acc.append(train_batch_acc)
                    if batch_count % PRETRAIN.SUMMARY_STEP == 0:
                        logging.info("global step: %d, epoch_%d, lr: %.5f, %d/%d: Train average loss: %.3f" %(global_step, epoch, lr, batch_count, trainset.batch_num, np.mean(train_epoch_loss)))

                logging.info('==========================================================')
                train_mean_acc = np.mean(train_epoch_acc)
                logging.info("epoch_%d: Train set(Aug) acc is: %.3f" % (epoch, train_mean_acc))
                if epoch % PRETRAIN.SAVE_EPOCH == 0:
                    saver.save(sess, PRETRAIN.SAVE_DIR + PRETRAIN.MODEL_TYPE + ".ckpt", global_step=global_step)

                if epoch % PRETRAIN.VALID_EPOCH == 0:
                    valid_epoch_acc = []
                    valid_epoch_loss = []
                    for image, label, batch_count in valset:
                        pre, valid_batch_loss = sess.run([predict, mean_loss],feed_dict={x1: image,
                                                                                         x2: label,
                                                                                         x3: False,
                                                                                         x4: 0,
                                                                                         x5: 0,})
                        batch_acc = np.mean(np.where(np.squeeze(pre) == label, 1, 0))
                        valid_epoch_acc.append(batch_acc)
                        valid_epoch_loss.append(valid_batch_loss)
                    valid_mean_acc = np.mean(valid_epoch_acc)
                    valid_mean_loss = np.mean(valid_epoch_loss)
                    logging.info("epoch_%d: Valid set average loss is: %.3f" % (epoch, valid_mean_loss))
                    logging.info("epoch_%d: Valid set acc is: %.3f" % (epoch, valid_mean_acc))
                    logging.info('==========================================================')





if __name__=='__main__':
    trianset = Dataset(PRETRAIN_SET)
    validset = Dataset(PREVAL_SET)
    train(trianset,validset)