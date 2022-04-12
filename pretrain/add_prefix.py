import tensorflow as tf
import os

old_model_path= 'E:\pycharm_project\segmentation_project\pretrain\\checkpoints\\mobilenetv3_large.ckpt-61115'
new_model_path= 'E:\pycharm_project\segmentation_project\pretrain\\mobilenetv3_add_prefix.ckpt'
add_prefix= 'FCN/'

def main():
    with tf.Session() as sess:
        new_var_list=[]
        for var_name, _ in tf.contrib.framework.list_variables(old_model_path):
            var = tf.contrib.framework.load_variable(old_model_path, var_name)
            new_name = var_name
            new_name = add_prefix + new_name
            print('Renaming %s to %s.' % (var_name, new_name))
            renamed_var = tf.Variable(var, name=new_name)
            new_var_list.append(renamed_var)

        print('starting to write new checkpoint !')
        saver = tf.train.Saver(var_list=new_var_list)
        sess.run(tf.global_variables_initializer())
        saver.save(sess, new_model_path)
        print("done !")

if __name__ == '__main__':
    main()
