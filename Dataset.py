import numpy as np
import cv2
import config as cfg
import random
from utils.image_preprocess import random_flip, random_shift, random_crop, random_rotate


class Dataset(object):
    def __init__(self, cfg_dict):
        self.cfg = cfg_dict
        self.list_dir = self.cfg.LIST_DIR
        self.batch_size = self.cfg.BATCH_SIZE
        self.img_size = self.cfg.IMG_SIZE
        self.is_aug = self.cfg.IS_AUG
        self.is_shuffle = self.cfg.IS_SHUFFLE
        self.aug_par = self.cfg.AUG_PAR

        with open(self.list_dir, 'r') as f:
            self.data_list = f.readlines()
        self.data_num = len(self.data_list)
        self.batch_num = int(np.ceil(self.data_num/self.batch_size))
        self.batch_count = 1

    def __iter__(self):
        return self

    def __next__(self):
        cur_img_batch = []
        cur_mask_batch = []

        if self.batch_count == 1 and self.is_shuffle:
            random.shuffle(self.data_list)

        start_index = (self.batch_count - 1) * self.batch_size
        for i in range(start_index, start_index+self.batch_size):
            cur_data_dir = self.data_list[i % self.data_num]
            cur_img, cur_mask = self.read_data(cur_data_dir)
            cur_img, cur_mask = self.preprocess(cur_img, cur_mask)
            cur_img_batch.append(cur_img)
            cur_mask_batch.append(cur_mask)
        self.batch_count = self.batch_count % self.batch_num + 1

        cur_img_batch = np.asarray(cur_img_batch, dtype=np.float32)
        cur_mask_batch = np.asarray(cur_mask_batch, dtype=np.float32)
        cur_mask_batch = np.expand_dims(cur_mask_batch, axis=-1)

        return cur_img_batch, cur_mask_batch


    def read_data(self, data_dir):
        img_dir, mask_dir = data_dir.strip("\n").split(" ")
        img = cv2.imread(img_dir,-1)
        mask = cv2.imread(mask_dir,-1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB

        return img, mask, self.batch_count


    def preprocess(self, img, mask):
        img = (img / 255 - self.cfg.NORM_MEAN) / self.cfg.NORM_STD
        if not self.is_aug:
            img = cv2.resize(img, (self.img_size,self.img_size))
            mask = cv2.resize(mask, (self.img_size,self.img_size))
        else:
            img = cv2.resize(img,(self.aug_par.CROP_IN, self.aug_par.CROP_IN))
            mask = cv2.resize(mask, (self.aug_par.CROP_IN, self.aug_par.CROP_IN))
            img, mask = random_crop(img, self.aug_par.CROP_IN, self.img_size, mask)
            img, mask = random_flip(img, self.aug_par.HFLIP_PROB, self.aug_par.VFLIP_PROB, mask)
            img, mask = random_shift(img, self.aug_par.X_SHIFT, self.aug_par.Y_SHIFT, mask)
            img, mask = random_rotate(img, self.aug_par.ROTATE_MIN, self.aug_par.ROTATE_MAX, mask)

        return img, mask

if __name__=='__main__':
    dataset = Dataset(cfg.VAL_SET)
    for img, mask in dataset:
        color_mask = np.zeros((mask[0].shape[0],mask[0].shape[1],3), dtype=np.float32)
        for i in range(mask[0].shape[0]):
            for j in range(mask[0].shape[1]):
                if mask[0][i][j]!=0:
                    color_mask[i][j]=[0,128,128]


        combined = cv2.addWeighted(img[0], 0.5, color_mask, 0.5, 0)
        cv2.imshow(',.', combined)
        cv2.waitKey(0)
