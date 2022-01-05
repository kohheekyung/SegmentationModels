from keras.layers import Layer, Input, Dropout, Conv2D, Activation, add, BatchNormalization, UpSampling2D, \
    Conv2DTranspose, Flatten, MaxPooling2D

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization, InputSpec
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.models import Model
from keras.engine.topology import Network
from keras import backend as K
import numpy as np
import time
import segmentation_models as sm
import tensorflow as tf
from base import loss_util
# from tensorflow.python.keras import layers
#
# import tensorflow.keras
class SegmentationModel():

    def __init__(self):

        # compile_losses = [self.dsc, self.jaccard]
        # compile_weights = [self.lambda_dsc, self.lambda_jaccard]

        #  Training parameters
        # self.lambda_dsc = 1.0
        # self.lambda_jaccard= 1.0
        self.compile_losses = [sm.losses.DiceLoss(), sm.losses.JaccardLoss(), self.segmentation_boundary_loss]
        self.compile_weights = [1, 1, 1]
        self.compile_metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

        self.learning_rate = 2e-4
        self.beta_1 = 0.5  # Adam parameter
        self.beta_2 = 0.999  # Adam parameter

        # Architecture parameters
        self.use_instance_normalization = True#False  # Use instance normalization or batch normalization
        self.use_dropout = True  # Dropout in residual blocks
        self.use_bias = True  # Use bias
        self.use_resize_convolution = False  # Resize convolution - instead of transpose convolution in deconvolution layers (uk) - can reduce checkerboard artifacts but the blurring might affect the cycle-consistency
        self.discriminator_sigmoid = True
        self.generator_residual_blocks = 9
        self.discriminator_layers = 5
        self.stride_2_layers = 3
        self.base_generator_filters = 32
        if self.use_instance_normalization:
            self.normalization = InstanceNormalization
        else:
            self.normalization = BatchNormalization
        self.activation = 'sigmoid'  # 'tanh'


        self.out_classes = 1
        self.input_channel = 1

        # Model parameters
        self.mode = 'ResNet' #  'preUNET' 'preLinknet' 'preFPN' 'prePSPNet'
        self.BACKBONE = 'efficientnetb3' #resnet34'  # 'efficientnetb3'
   

        self.input_shape_A = (1024, 1024,  self.input_channel)
        self.input_shape_B = (1024, 1024,  self.out_classes)

        self.base_model = None
        self.model = None

    def build_model(self):

        self.encoder_weight = None if self.input_shape_A[-1] == 1 else 'imagenet'
        # Optimizers
        self.opt= Adam(self.learning_rate, self.beta_1, self.beta_2)
        # Build generators

        if self.mode == 'ResNet':
            self.base_model = self.ResNet(self.input_shape_A, self.input_shape_B, name='Segmodel')
        elif self.mode == 'preUNET':
            self.base_model = sm.Unet(self.BACKBONE, encoder_weights=self.encoder_weight , encoder_freeze=True, classes=self.out_classes, activation=self.activation, input_shape=self.input_shape_A)
        elif self.mode == 'preFPN':
            self.base_model = sm.FPN(self.BACKBONE, encoder_weights=self.encoder_weight , classes=self.out_classes, activation=self.activation, input_shape=self.input_shape_A)
        elif self.mode == 'preLinknet':
            self.base_model = sm.Linknet(self.BACKBONE, encoder_weights=self.encoder_weight ,  classes=self.out_classes, activation=self.activation, input_shape=self.input_shape_A)
        elif self.mode == 'prePSPNet':
            self.base_model = sm.PSPNet(self.BACKBONE, encoder_weights=self.encoder_weight,  encoder_freeze=True, classes=self.out_classes, activation=self.activation, input_shape=self.input_shape_A,)

        # Compile full model
        input = Input(shape=self.input_shape_A, name='real_A')

        #input = Conv2D(3, (1, 1))(inp)  # map N channels data to 3 channels
        output = self.base_model(input)

        model_outputs = []
        for _ in range(len(self.compile_losses)):
            model_outputs.append(output)

        self.model = Model(inputs=[input], outputs =model_outputs)
        self.model.compile(optimizer=self.opt,
                             loss=self.compile_losses,
                             loss_weights=self.compile_weights)

        from contextlib import redirect_stdout
        with open('segmentation.txt', 'w') as f:
            with redirect_stdout(f):
                self.base_model.summary()

 # ===============================================================================
    # Architecture functions
    # First generator layer
    def c7Ak(self, x, k):
        x = Conv2D(filters=k, kernel_size=7, strides=1, padding='valid', use_bias=self.use_bias)(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    # Downsampling
    def dk(self, x, k):  # Should have reflection padding
        x = Conv2D(filters=k, kernel_size=3, strides=2, padding='same', use_bias=self.use_bias)(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    # Residual block
    def Rk(self, x0):
        k = int(x0.shape[-1])

        # First layer
        x = ReflectionPadding2D((1, 1))(x0)
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid', use_bias=self.use_bias)(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)

        if self.use_dropout:
            x = Dropout(0.5)(x)

        # Second layer
        x = ReflectionPadding2D((1, 1))(x)
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid', use_bias=self.use_bias)(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        # Merge
        x = add([x, x0])

        return x

    # Upsampling
    def uk(self, x, k):
        # (up sampling followed by 1x1 convolution <=> fractional-strided 1/2)
        if self.use_resize_convolution:
            x = UpSampling2D(size=(2, 2))(x)  # Nearest neighbor upsampling
            x = ReflectionPadding2D((1, 1))(x)
            x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid', use_bias=self.use_bias)(x)
        else:
            x = Conv2DTranspose(filters=k, kernel_size=3, strides=2, padding='same', use_bias=self.use_bias)(
                x)  # this matches fractionally stided with stride 1/2
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    # ===============================================================================
    # Models
    # # ResNet
    def ResNet(self, vol_shape_in, vol_shape_out, name=None):
        # Layer 1: Input
        input_vol = Input(shape=vol_shape_in)
        x = ReflectionPadding2D((3, 3))(input_vol)
        x = self.c7Ak(x, self.base_generator_filters)

        # Layer 2-3: Downsampling
        x = self.dk(x, 2 * self.base_generator_filters)
        x = self.dk(x, 4 * self.base_generator_filters)

        # Layers 4-12: Residual blocks
        #for _ in range(4, 4 + self.generator_residual_blocks):
        for _ in range(1, 4 + self.generator_residual_blocks):
            x = self.Rk(x)

        # Layer 13:14: Upsampling
        x = self.uk(x, 2 * self.base_generator_filters)
        x = self.uk(x, self.base_generator_filters)

        # Layer 15: Output
        x = ReflectionPadding2D((3, 3))(x)
        x = Conv2D(filters=vol_shape_out[-1], kernel_size=7, strides=1, padding='valid', use_bias=True)(x)
        x = Activation(self.activation)(x)

        return Model(inputs=input_vol, outputs=x, name=name)

    # Loss functions

    def lse(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.squared_difference(y_pred, y_true))
        return loss

    def cycle_loss(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.abs(y_pred - y_true))
        return loss

    # Mean p-norm error
    def mpd(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.abs(y_pred - y_true) ** 1.5)
        return loss

    # Mean p-norm error
    def mae(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.abs(y_pred - y_true))
        return loss

    def mse(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.abs(y_pred - y_true) ** 2.0)
        return loss

    def dsc(self, y_true, y_pred):
        smooth = 1.

        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

        intersection = tf.reduce_sum(y_true_f * y_pred_f)

        score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

        loss = 1 - score
        return loss

    def jaccard(self, y_true, y_pred):

        smooth = 1.
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

        intersection = tf.reduce_sum(tf.abs(y_true_f * y_pred_f))
        union = tf.reduce_sum(tf.abs(y_pred_f)) + tf.reduce_sum(tf.abs(y_true_f)) - intersection

        jac = (intersection + smooth) / (union + smooth)
        loss = 1 -jac
        return loss

    def gdl(self, y_true, y_pred):
        alpha = 2

        """
         Calculates the sum of GDL losses between the predicted and ground truth frames.
         This is the 3d version.
         @param gen_frames: The predicted frames at each scale.
         @param gt_frames: The ground truth frames at each scale
         @param alpha: The power to which each gradient term is raised.
         @return: The GDL loss for 3d. Dong
        """

        # calculate the loss for each scale
        scale_losses = []

        # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.
        pos = tf.constant(np.identity(1), dtype=tf.float32)
        neg = -1 * pos

        baseFilter = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]# 2x1x1x1
        filter_x = tf.expand_dims(baseFilter, 1)  # [-1, 1] # 2x1x1x1x1
        filter_y = tf.expand_dims(baseFilter, 0)  # [-1, 1] # 1x2x1x1x1
        filter_z = tf.expand_dims(tf.stack([neg, pos]), 0)  # [-1, 1] # 1x2x1x1
        filter_z = tf.expand_dims(filter_z, 0)  # [-1, 1] #1x1x2x1x1
        strides = [1, 1, 1, 1, 1]  # stride of (1, 1)
        padding = 'SAME'

        gen_dx = tf.abs(tf.nn.conv3d(y_pred, filter_x, strides, padding=padding))
        gen_dy = tf.abs(tf.nn.conv3d(y_pred, filter_y, strides, padding=padding))
        gen_dz = tf.abs(tf.nn.conv3d(y_pred, filter_z, strides, padding=padding))
        gt_dx = tf.abs(tf.nn.conv3d(y_true, filter_x, strides, padding=padding))
        gt_dy = tf.abs(tf.nn.conv3d(y_true, filter_y, strides, padding=padding))
        gt_dz = tf.abs(tf.nn.conv3d(y_true, filter_z, strides, padding=padding))

        grad_diff_x = tf.abs(gt_dx - gen_dx)
        grad_diff_y = tf.abs(gt_dy - gen_dy)
        grad_diff_z = tf.abs(gt_dz - gen_dz)

        loss = tf.reduce_mean((grad_diff_x ** alpha + grad_diff_y ** alpha + grad_diff_z ** alpha))

        # condense into one tensor and avg
        return loss

    def segmentation_boundary_loss(self, y_true, y_pred):
        """
        Paper Implemented : https://arxiv.org/abs/1905.07852
        Using Binary Segmentation mask, generates boundary mask on fly and claculates boundary loss.
        :param y_true:
        :param y_pred:
        :return:
        """
        y_pred_bd = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(1 - y_pred)
        y_true_bd = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(1 - y_true)
        y_pred_bd = y_pred_bd - (1 - y_pred)
        y_true_bd = y_true_bd - (1 - y_true)

        y_pred_bd_ext = MaxPooling2D((5, 5), strides=(1, 1), padding='same')(1 - y_pred)
        y_true_bd_ext = MaxPooling2D((5, 5), strides=(1, 1), padding='same')(1 - y_true)
        y_pred_bd_ext = y_pred_bd_ext - (1 - y_pred)
        y_true_bd_ext = y_true_bd_ext - (1 - y_true)

        P = K.sum(y_pred_bd * y_true_bd_ext) / K.sum(y_pred_bd) + 1e-7
        R = K.sum(y_true_bd * y_pred_bd_ext) / K.sum(y_true_bd) + 1e-7
        F1_Score = 2 * P * R / (P + R + 1e-7)
        # print(f'Precission: {P.eval()}, Recall: {R.eval()}, F1: {F1_Score.eval()}')
        loss = K.mean(1 - F1_Score)
        # print(f"Loss:{loss.eval()}")

        return loss

# reflection padding taken from
# https://github.com/fastai/courses/blob/master/deeplearning2/neural-style.ipynb
class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        size_increase = [0, 2*self.padding[0], 2*self.padding[1], 0]
        output_shape = list(s)

        for i in range(len(s)):
            if output_shape[i] == None:
                continue
            output_shape[i] += size_increase[i]

        return tuple(output_shape)

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')

    def get_config(self):
        config = {'padding': self.padding}
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
