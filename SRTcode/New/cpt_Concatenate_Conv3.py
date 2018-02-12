from __future__ import print_function
import os
import pickle
import numpy as np


import tensorflow as tf  #from V1707
import setproctitle  #from V1707
from keras import backend as K
K.set_image_data_format('channels_first')

config=tf.ConfigProto()  #from V1707
config.gpu_options.allow_growth=True  #from V1707
sess=tf.Session(config=config)  #from V1707
setproctitle.setproctitle('try@linziqian')  #from V1707

os.environ["DATAPATH"]='/home/stu/linziqian'


from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Reshape,Input,Activation,Dense,average,Concatenate,Add,Dropout,BatchNormalization
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
nb_epoch = 1000  # number of epoch at training stage
nb_epoch_cont = 100  # number of epoch at training (cont) stage
batch_size = 64  # batch size
T = 24  # number of time intervals in one day

lr = 0.0002  # learning rate
len_closeness = 3  # length of closeness dependent sequence
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
#m_factor = math.sqrt(1. * map_height * map_width / nb_area)
#print('factor: ', m_factor)
path_result = 'RET'
path_model = 'MODEL'

#set_trace()#!!!!!

if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)


#set_trace()#!!!!!
#load data
print("loading data...")
X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = BikeNYC.load_data( T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,preprocess_name='preprocessing.pkl', meta_data=True)
print("loading finish")

a=64

c_input=Input(shape=(6,16,8))
c_conv1=Convolution2D(filters=a,kernel_size=(3, 3),border_mode="same",activation='relu')(c_input)
c_conv1=BatchNormalization()(c_conv1)

p_input=Input(shape=(8,16,8))
p_conv1=Convolution2D(filters=a,kernel_size=(3, 3),border_mode="same",activation='relu')(p_input)
p_conv1=BatchNormalization()(p_conv1)

t_input=Input(shape=(8,16,8))
t_conv1=Convolution2D(filters=a,kernel_size=(3, 3),border_mode="same",activation='relu')(t_input)
t_conv1=BatchNormalization()(t_conv1)

merge=Concatenate(axis=1)([c_conv1,p_conv1,t_conv1])

cpt_conv1=Convolution2D(filters=a,kernel_size=(3, 3),border_mode="same",activation='relu')(merge)
cpt_conv1=BatchNormalization()(cpt_conv1)
cpt_conv2=Convolution2D(filters=a,kernel_size=(3, 3),border_mode="same",activation='relu')(cpt_conv1)
cpt_conv2=BatchNormalization()(cpt_conv2)
cpt_conv3=Convolution2D(filters=a,kernel_size=(3, 3),border_mode="same",activation='relu')(cpt_conv2)
cpt_conv3=BatchNormalization()(cpt_conv3)
cpt_conv=Convolution2D(filters=2,kernel_size=(3, 3),border_mode="same",activation='tanh')(cpt_conv3)

#sun_out=average(inputs=[cgconv2,p_conv2,t_conv2])

cpt_model=Model(inputs=[c_input,p_input,t_input],outputs=cpt_conv)
 
from keras.utils import plot_model
plot_model(cpt_model, to_file='best_Conv2.png', show_shapes=True)
#c_model=Model(inputs=c_input,outputs=c_conv2)

adam=Adam(lr)
cpt_model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
#c_model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
cpt_model.summary()
#c_model.summary()

model_checkpoint=ModelCheckpoint(
    filepath='Concatenate_Conv3.best.hdf5', 
    monitor='val_rmse', 
    verbose=1, 
    save_best_only=True, 
    save_weights_only=False, 
    mode='min', 
    period=1
)

history=cpt_model.fit(X_train[0:3],Y_train,nb_epoch=nb_epoch, verbose=1, callbacks=[model_checkpoint], batch_size=batch_size,validation_data=(X_test[0:3], Y_test))
output = open('.pkl','wb')
pickle.dump((history.history),output)
#cpt_model.fit(X_train,Y_train,nb_epoch=nb_epoch, verbose=1,callbacks=[model_checkpoint], batch_size=batch_size,validation_data=(X_test, Y_test))
