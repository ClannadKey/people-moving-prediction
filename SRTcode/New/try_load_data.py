from __future__ import print_function
import os
import cPickle as pickle
import numpy as np
import math


os.environ["CUDA_VISIBLE_DEVICES"]='3'
import tensorflow as tf  #from V1707
import setproctitle  #from V1707
from keras import backend as K
K.set_image_data_format('channels_first')

config=tf.ConfigProto()  #from V1707
#config.gpu_options.allow_growth=True  #from V1707
config.gpu_options.per_process_gpu_memory_fraction=0.7
sess=tf.Session(config=config)  #from V1707
#import keras.backend.tensorflow_backend as KTF
#KTF._set_session(tf.Session(config=config))
setproctitle.setproctitle('try@linziqian')  #from V1707


os.environ["DATAPATH"]='/home/stu/linziqian'


from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Reshape,Input,Activation,Dense,average,Concatenate,Add,Dropout,BatchNormalization,PReLU
from keras.models import Model
from keras.layers.convolutional import Convolution2D
from deepst.models.STResNet import stresnet
from deepst.config import Config
import deepst.metrics as metrics
from deepst.datasets import BikeNYC
np.random.seed(1337)  # for reproducibility
#from ipdb import set_trace
#set_trace()

# parameters
# data path, you may set your own data path with the global envirmental
# variable DATAPATH
DATAPATH = Config().DATAPATH
nb_epoch = 200  # number of epoch at training stage
nb_epoch_cont = 200  # number of epoch at training (cont) stage
batch_size = 32  # batch size
T = 24  # number of time intervals in one day

lr = 0.0002  # learning rate
len_closeness = 4  # length of closeness dependent sequence
len_period = 4  # length of peroid dependent sequence
len_trend = 4  # length of trend dependent sequence
nb_residual_unit = 4   # number of residual units

nb_flow = 2  # there are two types of flows: new-flow and end-flow
# divide data into two subsets: Train & Test, of which the test set is the
# last 10 days
days_test = 10
len_test = T * days_test
map_height, map_width = 16, 8  # grid size
# For NYC Bike data, there are 81 available grid-based areas, each of
# which includes at least ONE bike station. Therefore, we modify the final
# RMSE by multiplying the following factor (i.e., factor).
nb_area = 81
m_factor = math.sqrt(1. * map_height * map_width / nb_area)
print('factor: ', m_factor)
path_result = 'RET'
path_model = 'MODEL'

if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)
    
X_train0, Y_train0, X_test0, Y_test0, mmn, external_dim, timestamp_train, timestamp_test = BikeNYC.load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
        preprocess_name='preprocessing.pkl', meta_data=False)

X_train1, Y_train1, X_test1, Y_test1, mmn, external_dim, timestamp_train, timestamp_test = BikeNYC.load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test+1,
        preprocess_name='preprocessing.pkl', meta_data=False)

