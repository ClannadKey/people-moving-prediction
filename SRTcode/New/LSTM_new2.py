from __future__ import print_function
import os
import cPickle as pickle
import numpy as np
import math


os.environ["CUDA_VISIBLE_DEVICES"]='0'
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
from keras.layers import Reshape,Input,LSTM,Activation,Dense,average,Concatenate,Add,Dropout,BatchNormalization,PReLU
from keras.models import Model
from keras.layers.convolutional import Convolution2D
from deepst.models.STResNet import stresnet
from deepst.config import Config
import deepst.metrics as metrics
from deepst.datasets import BikeNYC
np.random.seed(1337)  # for reproducibility
from pdb import set_trace
#set_trace()

# parameters
# data path, you may set your own data path with the global envirmental
# variable DATAPATH
DATAPATH = Config().DATAPATH
nb_epoch = 200  # number of epoch at training stage
nb_epoch_cont = 200  # number of epoch at training (cont) stage
batch_size = 128  # batch size
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


#Conv2=np.zeros([8,8])
#LSTM2=np.zeros([8,8])
Conv2=np.load('Conv2.npy')
LSTM2=np.load('LSTM2.npy')

TT1=1
FF1=128
input1_128=Input(shape=[FF1])
cpt_conv1=PReLU()(input1_128)
cpt_conv1=BatchNormalization()(cpt_conv1)
cpt_conv1=Dropout(0.2)(cpt_conv1)
cpt_conv1=Dense(units=2,activation='tanh')(cpt_conv1)
model1=Model(inputs=input1_128,outputs=cpt_conv1)
model1.compile(loss='mse', optimizer=Adam(lr), metrics=[metrics.rmse])


print("loading data...")
X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = BikeNYC.load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
        preprocess_name='preprocessing.pkl', meta_data=False)
print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])

XX_train=np.load('XX_train.npy')
XXp_test=np.load('XXp_test.npy')

for X in range(8):
    for Y in range(8):
        if X<=0:
            continue
        if X==1 and Y<=5:
            continue
        
        XTRAIN=XX_train[:,:,X+8,Y]
        YTRAIN=Y_train[:,:,X+8,Y]
        XTEST=XXp_test[9:,:,X+8,Y]
        YTEST=Y_test[:,:,X+8,Y]
        
        print('XTRAIN.shape= ',XTRAIN.shape)
        print('YTRAIN.shape= ',YTRAIN.shape)
        print('XTEST.shape= ',XTEST.shape)
        print('YTEST.shape= ',YTEST.shape)
        
        F1='DENSE2.hdf5'
        
        model_checkpoint=ModelCheckpoint(
            filepath=F1,
            monitor='val_rmse',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            period=1
        )
        
        print('=' * 10)
        print("training model...")
        history = model1.fit(XTRAIN, YTRAIN,
                            nb_epoch=nb_epoch,
                            batch_size=batch_size,
                            validation_split=0.1,
                            callbacks=[model_checkpoint],
                            verbose=1)
        
        print('=' * 10)
        print('evaluating using the model that has the best loss on the valid set')
        model1.load_weights(F1)
        score = model1.evaluate(XTRAIN, YTRAIN, batch_size=YTRAIN.shape[0] // 48, verbose=0)
        print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
                  (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2. * m_factor))
        score = model1.evaluate(XTEST, YTEST, batch_size=YTEST.shape[0], verbose=0)
        print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
                  (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2. * m_factor))

        Conv2[X,Y]=score[1] * (mmn._max - mmn._min) / 2. * m_factor


        TT2=10
        FF2=128
        
        input2_128=Input(shape=(TT2,FF2))
        cpt_conv2=PReLU()(input2_128)
        cpt_conv2=BatchNormalization()(cpt_conv2)
        cpt_conv2=LSTM(units=2,activation='tanh', recurrent_activation='hard_sigmoid',return_sequences=False)(cpt_conv2)
        output2=Dense(units=2)(cpt_conv2)
        model2=Model(inputs=input2_128,outputs=output2)
        model2.compile(loss='mse', optimizer=Adam(lr), metrics=[metrics.rmse])
        
        Xtrain=XX_train[:,:,X+8,Y].reshape([-1,1,FF2])
        XTRAIN=Xtrain[:-9]
        for i in range(8):
            XTRAIN=np.concatenate((XTRAIN,Xtrain[i+1:-(8-i)]),axis=1)
        XTRAIN=np.concatenate((XTRAIN,Xtrain[9:]),axis=1)
        
        Xptest=XXp_test[:,:,X+8,Y].reshape([-1,1,FF2])
        XTEST=Xptest[:-9]
        for i in range(8):
            XTEST=np.concatenate((XTEST,Xptest[i+1:-(8-i)]),axis=1)
        XTEST=np.concatenate((XTEST,Xptest[9:]),axis=1)
        
        YTRAIN=Y_train[9:,:,X+8,Y].reshape([-1,2])
        YTEST=Y_test[:,:,X+8,Y].reshape([-1,2])
        
        print('XTRAIN.shape= ',XTRAIN.shape)
        print('YTRAIN.shape= ',YTRAIN.shape)
        print('XTEST.shape= ',XTEST.shape)
        print('YTEST.shape= ',YTEST.shape)

        F2='LSTM2.hdf5'
        
        model_checkpoint=ModelCheckpoint(
            filepath=F2,
            monitor='val_rmse',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            period=1
        )
        
        print('=' * 10)
        print("training model...")
        history = model2.fit(XTRAIN, YTRAIN,
                            nb_epoch=nb_epoch,
                            batch_size=batch_size,
                            validation_split=0.1,
                            callbacks=[model_checkpoint],
                            verbose=1)
        
        print('=' * 10)
        print('evaluating using the model that has the best loss on the valid set')
        model2.load_weights(F2)
        score = model2.evaluate(XTRAIN, YTRAIN, batch_size=YTRAIN.shape[0] // 48, verbose=0)
        print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
                  (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2. * m_factor))
        score = model2.evaluate(XTEST, YTEST, batch_size=YTEST.shape[0], verbose=0)
        print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
                  (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2. * m_factor))
        
        LSTM2[X,Y]=score[1] * (mmn._max - mmn._min) / 2. * m_factor
       
        print(X+8,Y)
        print('Conv2',Conv2[X,Y])
        print('LSTM2',LSTM2[X,Y])
        np.save('Conv2.npy',Conv2)
        np.save('LSTM2.npy',LSTM2)
