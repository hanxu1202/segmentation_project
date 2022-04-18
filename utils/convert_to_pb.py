import tensorflow as tf
from tensorflow.python.framework import graph_util

meta_dir = 'E:\pycharm_project\segmentation_project\checkpoints\Deeplabv3p\\Deeplabv3p_mobilenetv2_16s_3c.ckpt-29100.meta'
ckpt_dir = 'E:\pycharm_project\segmentation_project\checkpoints\Deeplabv3p\\Deeplabv3p_mobilenetv2_16s_3c.ckpt-29100'
out_pb_dir = 'E:\pycharm_project\segmentation_project\checkpoints\Deeplabv3p\\Deeplabv3p_mobilenetv2_16s_3c.pb'

g1 = tf.Graph()
with g1.as_default():
    sess = tf.Session(graph=g1)
    loader = tf.train.import_meta_graph(meta_dir)
    loader.restore(sess, ckpt_dir)

    for n in g1.as_graph_def().node:
        print(n.name)
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['Deeplabv3p/decoder/logits_output'])
    with tf.gfile.FastGFile(out_pb_dir, mode='wb') as f:
        f.write(constant_graph.SerializeToString())