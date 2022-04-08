import numpy as np
import cv2

def random_flip(img, h_flip, v_flip, mask=None):
    h = np.random.rand(1)
    v = np.random.rand(1)
    if h <= h_flip:
        img = cv2.flip(img, 1)
        if mask is not None:
            mask = cv2.flip(mask, 1)
    if v <= v_flip:
        img = cv2.flip(img, 0)
        if mask is not None:
            mask = cv2.flip(mask,0)
    if mask is not None:
        return img, mask
    else:
        return img


def random_rotate(img, min_degree, max_degree, mask=None):
    h, w = img.shape[0], img.shape[1]
    center = [int(w / 2), int(h / 2)]
    angle = np.random.uniform(min_degree, max_degree)
    matrix = cv2.getRotationMatrix2D(center, angle, 1)
    output_size = [w, h]
    img = cv2.warpAffine(img, matrix, output_size, borderMode=cv2.BORDER_REFLECT)
    if mask is not None:
        mask = cv2.warpAffine(mask, matrix, output_size, borderMode=cv2.BORDER_REFLECT)
        return img, mask
    else:
        return img

def random_crop(img, shape_in, shape_out, mask=None):
    crop_size = shape_in-shape_out
    left = np.random.randint(0, int(crop_size))
    bottom = np.random.randint(0, int(crop_size))

    img = img[bottom:shape_in-(crop_size-bottom), left:shape_in-(crop_size-left)]
    if mask is not None:
        mask = mask[bottom:shape_in-(crop_size-bottom), left:shape_in-(crop_size-left)]
        return img, mask
    else:
        return img


def random_shift(img, x_shift, y_shift, mask=None):
    h, w = img.shape[:2]
    x_shift = np.random.randint(-int(w * x_shift), int(w * x_shift))
    y_shift = np.random.randint(-int(h * y_shift), int(h * y_shift))
    matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    output_size = [w, h]
    img = cv2.warpAffine(img, matrix, output_size, borderMode=cv2.BORDER_REFLECT)
    if mask is not None:
        mask = cv2.warpAffine(mask, matrix, output_size, borderMode=cv2.BORDER_REFLECT)
        return img, mask
    else:
        return img

