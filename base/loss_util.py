import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from base import Loss

SMOOTH = 1e-5


def segmentation_boundary_loss(y_true, y_pred):
    """
    Paper Implemented : https://arxiv.org/abs/1905.07852
    Using Binary Segmentation mask, generates boundary mask on fly and claculates boundary loss.
    :param y_true:
    :param y_pred:
    :return:
    """
    y_pred_bd = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(1 - y_pred)
    y_true_bd = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(1 - y_true)
    y_pred_bd = y_pred_bd - (1 - y_pred)
    y_true_bd = y_true_bd - (1 - y_true)

    y_pred_bd_ext = layers.MaxPooling2D((5, 5), strides=(1, 1), padding='same')(1 - y_pred)
    y_true_bd_ext = layers.MaxPooling2D((5, 5), strides=(1, 1), padding='same')(1 - y_true)
    y_pred_bd_ext = y_pred_bd_ext - (1 - y_pred)
    y_true_bd_ext = y_true_bd_ext - (1 - y_true)

    P = K.sum(y_pred_bd * y_true_bd_ext) / K.sum(y_pred_bd) + 1e-7
    R = K.sum(y_true_bd * y_pred_bd_ext) / K.sum(y_true_bd) + 1e-7
    F1_Score = 2 * P * R / (P + R + 1e-7)
    # print(f'Precission: {P.eval()}, Recall: {R.eval()}, F1: {F1_Score.eval()}')
    loss = K.mean(1 - F1_Score)
    # print(f"Loss:{loss.eval()}")
    return loss


class BoundaryLoss(Loss):

    def __init__(self, class_weights=None, class_indexes=None, per_image=False, smooth=SMOOTH):
        super().__init__(name='boundary_loss')
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.per_image = per_image
        self.smooth = smooth

    def __call__(self, gt, pr):
        return segmentation_boundary_loss(gt, pr)
