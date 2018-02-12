
from __future__ import print_function
import os
import cPickle as pickle
import numpy as np
import math


os.environ["CUDA_VISIBLE_DEVICES"]='3'
import tensorflow as tf  #from V1707
import setproctitle  #from V1707
config=tf.ConfigProto()  #from V1707
#config.gpu_options.allow_growth=True  #from V1707
config.gpu_options.per_process_gpu_memory_fraction=0.4
sess=tf.Session(config=config)  #from V1707
#import keras.backend.tensorflow_backend as KTF
#KTF._set_session(tf.Session(config=config))
setproctitle.setproctitle('try@linziqian')  #from V1707


os.environ["DATAPATH"]='/home/stu/linziqian'

from keras import backend as K
K.set_image_data_format('channels_first')
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

a=64
Q=2
drop=0.2
c_input=Input(shape=(8,16,8))
p_input=Input(shape=(8,16,8))
t_input=Input(shape=(8,16,8))
    
merge=Concatenate(axis=1)([c_input,p_input,t_input])
cpt_conv1=Convolution2D(filters=int(Q*a),kernel_size=(3, 3),border_mode="same")(merge)

cpt_conv2=PReLU()(cpt_conv1)
cpt_conv2=BatchNormalization()(cpt_conv2)
cpt_conv2=Dropout(drop)(cpt_conv2)
cpt_conv2=Convolution2D(filters=int(Q*a),kernel_size=(3, 3),border_mode="same")(cpt_conv2)
cpt_conv2=Add()([cpt_conv1,cpt_conv2])
    
cpt_conv3=PReLU()(cpt_conv2)
cpt_conv3=BatchNormalization()(cpt_conv3)
cpt_conv3=Dropout(drop)(cpt_conv3)
cpt_conv3=Convolution2D(filters=int(Q*a),kernel_size=(3, 3),border_mode="same")(cpt_conv3)
cpt_conv3=Add()([cpt_conv2,cpt_conv3])
    
cpt_conv4=PReLU()(cpt_conv3)
cpt_conv4=BatchNormalization()(cpt_conv4)
cpt_conv4=Dropout(drop)(cpt_conv4)
cpt_conv4=Convolution2D(filters=int(Q*a),kernel_size=(3, 3),border_mode="same")(cpt_conv4)
cpt_conv4=Add()([cpt_conv3,cpt_conv4])

cpt_conv=PReLU(shared_axes=[2,3])(cpt_conv4)
cpt_conv=BatchNormalization()(cpt_conv)
cpt_conv=Dropout(drop)(cpt_conv)
cpt_conv=Convolution2D(filters=2,kernel_size=(1, 1),border_mode="same",activation='tanh')(cpt_conv)

model1=Model(inputs=[c_input,p_input,t_input],outputs=cpt_conv)
adam=Adam(lr)
model1.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
model1.summary()

model2=Model(inputs=[c_input,p_input,t_input],outputs=cpt_conv4)

from keras.utils import plot_model
plot_model(model1, to_file='model1.png', show_shapes=True)

# load data
print("loading data...")
X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = BikeNYC.load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
        preprocess_name='preprocessing.pkl', meta_data=False)

Xp_train, Yp_train, Xp_test, Yp_test, mmn, external_dim, timestamp_train, timestamp_test = BikeNYC.load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test+9,
        preprocess_name='preprocessing.pkl', meta_data=False)

print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])

F='128.hdf5'

model1_checkpoint=ModelCheckpoint(
    filepath=F,
    monitor='val_rmse',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    period=1
)

print('=' * 10)
print("training model...")
history = model1.fit(X_train, Y_train,
                    nb_epoch=nb_epoch,
                    batch_size=batch_size,
                    validation_split=0.1,
                    callbacks=[model1_checkpoint],
                    verbose=1)

print('=' * 10)
print('evaluating using the model that has the best loss on the valid set')
model1.load_weights(F)
score = model1.evaluate(X_train, Y_train, batch_size=Y_train.shape[0] // 48, verbose=0)
print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2. * m_factor))
score = model1.evaluate(X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2. * m_factor))

XX_train=model2.predict(X_train)
XXp_test=model2.predict(Xp_test)
np.save('XX_train.npy',XX_train)
np.save('XXp_test.npy',XXp_test)

