from easydict import EasyDict as edict

# use for debugging
TEST_SET = edict({
    'LIST_DIR': 'E:\COCO2017/test.txt',
    'BATCH_SIZE': 1,
    'IMG_SIZE': 224,
    'NORM_MEAN': [0.485, 0.456, 0.406],
    'NORM_STD': [0.229, 0.224, 0.225],
    'IS_AUG': False,
    'IS_SHUFFLE': False,
    'AUG_PAR': None
})

TRAIN_SET = edict({
    'LIST_DIR': 'E:\dogcat_seg\\train.txt',
    'BATCH_SIZE': 16,
    'IMG_SIZE': 224,
    'NORM_MEAN': [0.485, 0.456, 0.406],
    'NORM_STD': [0.229, 0.224, 0.225],
    'IS_AUG': True,
    'IS_SHUFFLE': True,
    'AUG_PAR': {'CROP_IN':256,
               'HFLIP_PROB':0.4,
               'VFLIP_PROB':0.4,
               'ROTATE_MIN':-20,
               'ROTATE_MAX':20,
               'X_SHIFT':0.2,
               'Y_SHIFT':0.2}
})


VAL_SET = edict({
    'LIST_DIR': 'E:\dogcat_seg\\val.txt',
    'BATCH_SIZE': 16,
    'IMG_SIZE': 224,
    'NORM_MEAN': [0.485, 0.456, 0.406],
    'NORM_STD': [0.229, 0.224, 0.225],
    'IS_AUG': False,
    'IS_SHUFFLE': False,
    'AUG_PAR': None
})


TRAIN = edict({
    'START_EPOCH': 1,
    'TOTAL_EPOCH': 200,
    'PRETRAINED_BACKBONE':'E:\pycharm_project\segmentation_project\\checkpoints\\pretrained_backbones\\mobilenetv3_add_prefix.ckpt' ,
    'RESUME_CKPT': None,
    'NUM_CLASSES': 3,
    'INIT_LR': 0.001,
    'LR_DECAY_STEP': 2500,
    'LR_DECAY_RATE': 0.9,
    'SUMMARY_STEP': 10,
    'BACKBONE_TYPE': 'mobilenetv3_large',
    'MODEL_TYPE': 'FCN',
    'DS_FEATURE': 4,
    'SAVE_DIR': 'E:\pycharm_project\segmentation_project\\checkpoints\\',
    'SAVE_EPOCH': 1,
    'VALID_EPOCH': 1,
})


PREDICT = edict({
    'IMAGE_DIR': 'E:\pycharm_project\segmentation_project\\test_img\\1.jpg',
    'IMG_SIZE': 224,
    'NORM_MEAN': [0.485, 0.456, 0.406],
    'NORM_STD': [0.229, 0.224, 0.225],
    'MODEL_CKPT': None,
    'NUM_CLASSES': 3,
    'BACKBONE_TYPE': 'mobilenetv3_large',
    'MODEL_TYPE': 'FCN',
    'DS_FEATURE': 4,
    'MODEL_CKPT': 'E:\pycharm_project\segmentation_project\checkpoints\\FCN4s_mobilenetv3_large\\FCN4s_mobilenetv3_large.ckpt-77600'

})
