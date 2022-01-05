import os
import tensorflow as tf
import keras.backend as K
import numpy as np

from train import *
from data import *
from model import *
#
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)  # Select GPU device±

# data configuration
data = data()
data.data_root = 'insert data root directory'
data.volume_folder = 'insert data directory'
data.preprocess = False
data.inputA_size = (256,256)# (128,128)# (1024, 1024) # 434 636
data.inputA_channel = 1
data.inputB_size = (256,256)#(128,128)#(1024, 1024)
data.inputB_channel = 1
data.load_datalist()

# model configuration
model = SegmentationModel()
model.input_shape_A = data.inputA_size + (data.inputA_channel, )
model.input_shape_B = data.inputB_size + (data.inputB_channel, )
model.build_model()


train = train()
train.model_dir = 'model will be saved under this directory'
train.iter_perEpoch = 1600
#train.retrain_epoch = 5
#train.retrain_path ='model path to retrain'
train.make_folders()
train.load_data(data)
train.load_model(model)
train.train()
#train.retrain()