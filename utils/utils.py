import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

def conmpute_confusion_matrix(pre, gt, num_classes):
    """
        生成混淆矩阵
        pre 是形状为(HxW,)的预测值
        gt 是形状为(HxW,)的真实值
        num_classes 是类别数
    """
    # 确保a和b在0~n-1的范围内，k是(HxW,)的True和False数列
    k = (pre >= 0) & (pre < num_classes)
    # 这句会返回混淆矩阵，具体在下面解释
    return np.bincount(num_classes * pre[k].astype(int) + gt[k], minlength=num_classes ** 2).reshape(num_classes, num_classes)


def compute_miou(pre, gt, num_classes):
    confusion_matrix = np.zeros(shape=[num_classes,num_classes],dtype=np.int32)
    for i in range(pre.shape[0]):
        confusion_matrix += conmpute_confusion_matrix(pre, gt, num_classes)
    iou_classes = np.diag(confusion_matrix) / (confusion_matrix.sum(1) + confusion_matrix.sum(0) - np.diag(confusion_matrix) + 0.000001)
    miou = np.mean(iou_classes)
    return miou


def draw_colored_mask(img, mask, color_map):
    color_mask = np.zeros(shape=[img.shape[0], img.shape[1], img.shape[2]], dtype=np.uint8)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            color_mask[i][j][:] = color_map[mask[i][j]][:]
    masked_img = cv2.addWeighted(img, 1, color_mask, 0.8, 0)
    return masked_img

def CRF_process(ori_prob, ori_img, num_classes, iteration=5):
    # CRF
    d = dcrf.DenseCRF2D(ori_img.shape[1], ori_img.shape[0], num_classes) # w, h, n_classes

    # get unary
    prob = np.squeeze(ori_prob)
    prob = cv2.resize(prob, (ori_img.shape[1], ori_img.shape[0])) # w, h
    prob = prob.transpose(2, 0, 1)
    U = unary_from_softmax(prob)
    U = np.ascontiguousarray(U)
    d.setUnaryEnergy(U)

    # PAIRWISE
    d.addPairwiseGaussian(sxy=3, compat=3)
    #d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=im, compat=10)

    # inference
    Q = d.inference(iteration)

    # map
    map = np.argmax(Q, axis=0).reshape((ori_img.shape[0], ori_img.shape[1]))
    return map

if __name__=='__main__':
    img = cv2.imread('../test_img.jpg', -1)
    mask = cv2.imread('../test_mask.png', -1)
    color_map = {0: [0, 0, 0],
                 1: [0, 128, 128],
                 2: [128, 0, 128]}

    result = draw_colored_mask(img, mask, color_map)
    cv2.imshow('dad', result)
    cv2.waitKey(0)