from easydict import EasyDict as edict

# use for debugging
VAL_SET = edict({
    'LIST_DIR': 'E:\COCO2017/test.txt',
    'BATCH_SIZE': 5,
    'IMG_SIZE': 224,
    'NORM_MEAN': [0.485, 0.456, 0.406],
    'NORM_STD': [0.229, 0.224, 0.225],
    'IS_AUG': False,
    'IS_SHUFFLE': False,
    'AUG_PAR': None
})

PRETRAIN_SET = edict({
    'LIST_DIR': 'E:\dogcat\\train.txt',
    'BATCH_SIZE': 32,
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

PREVAL_SET = edict({
    'LIST_DIR': 'E:\dogcat\\val.txt',
    'BATCH_SIZE': 32,
    'IMG_SIZE': 224,
    'NORM_MEAN': [0.485, 0.456, 0.406],
    'NORM_STD': [0.229, 0.224, 0.225],
    'IS_AUG': False,
    'IS_SHUFFLE': False,
    'AUG_PAR': None
})

PRETRAIN = edict({
    'START_EPOCH': 1,
    'TOTAL_EPOCH': 50,
    'RESUME_CKPT': 'E:\pycharm_project\segmentation_project\pretrain\checkpoints\\mobilenetv3_large.ckpt-14380',
    'NUM_CLASSES': 3,
    'INIT_LR': 0.001,
    'LR_DECAY_STEP': 1000,
    'LR_DECAY_RATE': 0.9,
    'SUMMARY_STEP': 10,
    'NET_TYPE':'mobilenetv3_large',
    'SAVE_DIR': 'E:\pycharm_project\segmentation_project\pretrain\checkpoints\\',
    'SAVE_EPOCH': 1,
    'VALID_EPOCH': 1,
})


TRAIN_SET = edict({
    'LIST_DIR': 'E:\COCO2017/coco_train2017_catdog.txt',
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
    'LIST_DIR': 'E:\COCO2017/coco_val2017_catdog.txt',
    'BATCH_SIZE': 16,
    'IMG_SIZE': 224,
    'NORM_MEAN': [0.485, 0.456, 0.406],
    'NORM_STD': [0.229, 0.224, 0.225],
    'IS_AUG': False,
    'IS_SHUFFLE': False,
    'AUG_PAR': None
})


