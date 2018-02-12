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
from keras.callbacks import ModelCheckpoint
from keras.layers import Input,Concatenate,LSTM,Reshape,Dense,Conv2D
from keras.models import Model
from deepst.config import Config
import deepst.metrics as metrics
from deepst.datasets import BikeNYC 
np.random.seed(1337)  # for reproducibility
#from ipdb import set_trace
#set_trace()



# parameters
# data path, you may set your own data path with the global envirmental
# variable DATAPATH
#DATAPATH = Config().DATAPATH
nb_epoch = 1000  # number of epoch at training stage
nb_epoch_cont = 100  # number of epoch at training (cont) stage
batch_size = 256  # batch size
T = 24  # number of time intervals in one day

lr = 0.0001  # learning rate
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


c_in0=Input(shape=(3,2,16,8))
c_in0_reshape=Reshape((3,2*16*8))(c_in0)
c_in1=LSTM(units=256,return_sequences=True)(c_in0_reshape)
c_in2=LSTM(units=256,return_sequences=False)(c_in1)
c_in3=Dense(256)(c_in2)
c_out=Reshape((2,16,8))(c_in3)


c_cl=Model(inputs=c_in0,outputs=c_out)

c_cl.summary()

print("loading data...")
X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = BikeNYC.load_data(T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,preprocess_name='preprocessing.pkl', meta_data=True)
print("loading finished")
print("reshape data...")
X_train_c=X_train[0].reshape(-1,3,2,16,8)
X_train_c=np.concatenate((X_train_c[:,2,:,:,:],X_train_c[:,1,:,:,:],X_train_c[:,0,:,:,:]),axis=1)
X_train_c=X_train_c.reshape(-1,3,2,16,8)

X_train_p=X_train[1].reshape(-1,4,2,16,8)
X_train_p=np.concatenate((X_train_p[:,3,:,:,:],X_train_p[:,2,:,:,:],X_train_p[:,1,:,:,:],X_train_p[:,0,:,:,:]),axis=1)
X_train_p=X_train_p.reshape(-1,4,2,16,8)

X_train_t=X_train[2].reshape(-1,4,2,16,8)
X_train_t=np.concatenate((X_train_t[:,3,:,:,:],X_train_t[:,2,:,:,:],X_train_t[:,1,:,:,:],X_train_t[:,0,:,:,:]),axis=1)
X_train_t=X_train_t.reshape(-1,4,2,16,8)

myX_train=[X_train_c,X_train_p,X_train_t]

X_test_c=X_test[0].reshape(-1,3,2,16,8)
X_test_c=np.concatenate((X_test_c[:,2,:,:,:],X_test_c[:,1,:,:,:],X_test_c[:,0,:,:,:]),axis=1)
X_test_c=X_test_c.reshape(-1,3,2,16,8)

X_test_p=X_test[1].reshape(-1,4,2,16,8)
X_test_p=np.concatenate((X_test_p[:,3,:,:,:],X_test_p[:,2,:,:,:],X_test_p[:,1,:,:,:],X_test_p[:,0,:,:,:]),axis=1)
X_test_p=X_test_p.reshape(-1,4,2,16,8)

X_test_t=X_test[2].reshape(-1,4,2,16,8)
X_test_t=np.concatenate((X_test_t[:,3,:,:,:],X_test_t[:,2,:,:,:],X_test_t[:,1,:,:,:],X_test_t[:,0,:,:,:]),axis=1)
X_test_t=X_test_t.reshape(-1,4,2,16,8)

myX_test=[X_test_c,X_test_p,X_test_t]
print("reshape finished")


adam=Adam(lr)
c_cl.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])

model_checkpoint=ModelCheckpoint(
    filepath='c_LSTM2_Dense1.best.hdf5',
    monitor='val_rmse',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    period=1
)

c_cl.compile(loss='mse', optimizer=Adam(lr*10), metrics=[metrics.rmse])
history=c_cl.fit(myX_train[0],Y_train,nb_epoch=200, verbose=1, callbacks=[model_checkpoint], batch_size=batch_size,validation_data=(myX_test[0], Y_test))
c_cl.compile(loss='mse', optimizer=Adam(lr), metrics=[metrics.rmse])
history=c_cl.fit(myX_train[0],Y_train,nb_epoch=nb_epoch, verbose=1, callbacks=[model_checkpoint], batch_size=batch_size,validation_data=(myX_test[0], Y_test))

output = open('c_LSTM2_Dense1.pkl','wb')
pickle.dump((history.history),output)
