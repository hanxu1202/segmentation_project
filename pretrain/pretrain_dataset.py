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
        cur_label_batch = []

        if self.batch_count == 1 and self.is_shuffle:
            random.shuffle(self.data_list)

        if self.batch_count <= self.batch_num:
            start_index = (self.batch_count - 1) * self.batch_size
            for i in range(start_index, start_index+self.batch_size):
                cur_data_dir = self.data_list[i % self.data_num]
                cur_img, cur_label = self.read_data(cur_data_dir)
                cur_img = self.preprocess(cur_img)
                cur_img_batch.append(cur_img)
                cur_label_batch.append(cur_label)
            self.batch_count = self.batch_count + 1

            cur_img_batch = np.asarray(cur_img_batch, dtype=np.float32)
            cur_label_batch = np.asarray(cur_label_batch, dtype=np.int32)
            #cur_label_batch = np.expand_dims(cur_label_batch, axis=-1)

            return cur_img_batch, cur_label_batch, self.batch_count-1
        else:
            self.batch_count = 1
            raise StopIteration


    def read_data(self, data_dir):
        img_dir, label = data_dir.strip("\n").split(" ")
        label = np.float(label)
        img = cv2.imread(img_dir,-1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
        return img, label


    def preprocess(self, img):
        img = (img / 255 - self.cfg.NORM_MEAN) / self.cfg.NORM_STD
        if not self.is_aug:
            img = cv2.resize(img, (self.img_size, self.img_size))
        else:
            img = cv2.resize(img,(self.aug_par.CROP_IN, self.aug_par.CROP_IN))
            img = random_crop(img, self.aug_par.CROP_IN, self.img_size)
            img = random_flip(img, self.aug_par.HFLIP_PROB, self.aug_par.VFLIP_PROB)
            img = random_shift(img, self.aug_par.X_SHIFT, self.aug_par.Y_SHIFT)
            img = random_rotate(img, self.aug_par.ROTATE_MIN, self.aug_par.ROTATE_MAX)

        return img

if __name__=='__main__':
    dataset = Dataset(cfg.PRETRAIN_SET)
    for img, label in dataset:
        print(img.shape, label.shape)
