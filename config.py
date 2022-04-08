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


