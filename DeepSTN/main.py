#from __future__ import print_function
from Data.lzq_read_data_time_poi import lzq_load_data
from Data.read_taxibj import load_taxibj
from keras.callbacks import ModelCheckpoint, EarlyStopping
#import cPickle as pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Spatial-Temporal Dynamic Network')
parser.add_argument('--dataset', type=str, default='BikeNYC', help='BikeNYC, TaxiNYC or TaxiBJ')
parser.add_argument('--device', type=str, default='0')
args = parser.parse_args()
print(args)

NO=4
#for reproduction
seed=1
for i in range(NO):
    seed=seed*10+7
seed=seed*10+7
np.random.seed(seed)
#from ipdb import set_trace
#set_trace()

#for GPU in Lab
device = args.device

import os

os.environ["CUDA_VISIBLE_DEVICES"] = device
import setproctitle  #from V1707
setproctitle.setproctitle('Comprison Start! @ ZiqianLin')  #from V1707

from keras import backend as K
K.set_image_data_format('channels_first')


#hyperparameters
epoch = 350  # number of epoch at training stage
batch_size = 8  # batch size
lr = 0.0002  # learning rate

H,W,channel = 10,20,2   # grid size

T = 24*2  # number of time intervals in one day

len_closeness = 3  # length of closeness dependent sequence
len_period = 4  # length of peroid dependent sequence
len_trend = 4  # length of trend dependent sequence

T_closeness,T_period,T_trend=1,T,T*7

# last 7 days for testing data
days_test = 20
len_test = T * days_test

#the number of repetition and if retrain the model
iterate_num=1


X10=1   #DSTN+ResPlus
train10=1   #DSTN+ResPlus

class CustomStopper(EarlyStopping):
    # add argument for starting epoch
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', start_epoch=40):
        super().__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)

#DSTN+ResPlus
if X10:
    setproctitle.setproctitle('BJMobile DSTN+ResPlus @ ZiqianLin')  #from V1707
    from DeepSTN_network.DeepSTN_net import DeepSTN
    
    
    if args.dataset == 'BikeNYC':
        X_train, T_train, Y_train, X_test, T_test, Y_test, MM = \
            lzq_load_data('Data/bike_flow.npy', len_test, len_closeness, len_period, len_trend, T_closeness,
                          T_period, T_trend)
    elif args.dataset == 'TaxiNYC':
        X_train, T_train, Y_train, X_test, T_test, Y_test, MM = \
            lzq_load_data('Data/taxi_flow.npy', len_test, len_closeness, len_period, len_trend, T_closeness,
                          T_period, T_trend)
    elif args.dataset == 'TaxiBJ':
        X_train, T_train, Y_train, X_test, T_test, Y_test, MM, Max, Min = \
            load_taxibj('Data/BJ13_M32x32_T30_InOut.h5', len_test, len_closeness, len_period, len_trend, T_closeness,
                          T_period, T_trend)
        H, W, channel = 32, 32, 2
    else:
        raise ValueError('dataset does not exist')
                          
    X_train = np.concatenate((X_train[0], X_train[1], X_train[2]), axis=1)
    X_test = np.concatenate((X_test[0], X_test[1], X_test[2]), axis=1)

    pre_F=64
    conv_F=64
    R_N=2

    is_plus=True
    plus=8
    rate=1

    is_pt=False
    P_N=9
    T_F=7*8
    PT_F=9

    drop=0.1

    import time
    count=0
    count_sum=iterate_num
    
    iterate_loop=np.arange(iterate_num)+1+iterate_num*(NO-1)
    
    Pickup_RMSE = np.zeros([iterate_num, 1])
    Pickup_MAE = np.zeros([iterate_num, 1])
    Dropoff_RMSE = np.zeros([iterate_num, 1])
    Dropoff_MAE = np.zeros([iterate_num, 1])
    
    for iterate_index in range(iterate_num):
        count=count+1      
        iterate=iterate_loop[iterate_index]
        
        print("***** conv_model *****")
        model=DeepSTN(H=H,W=W,channel=channel,
                      c=len_closeness,p=len_period,                
                      pre_F=pre_F,conv_F=conv_F,R_N=R_N,    
                      is_plus=is_plus,
                      plus=plus,rate=rate,     
                      is_pt=is_pt,P_N=P_N,T_F=T_F,PT_F=PT_F,T=T,     
                      drop=drop)            
        
        file_conv='Exps/DeepSTN_eval_model_'+args.dataset+'.hdf5'
        #train conv_model
        if train10:
            
            model_checkpoint=ModelCheckpoint(
                    filepath=file_conv,
                    monitor='val_pickup_rmse',
                    verbose=1,
                    save_best_only=True,
                    save_weights_only=True,
                    mode='min',
                    period=1
                )
            
            stop = CustomStopper(monitor='val_pickup_rmse', min_delta=0, patience=5, verbose=0, mode='min', start_epoch=40)
            
            train_start=time.time() 
            print('=' * 10)
            print("***** training conv_model *****")
            history = model.fit(X_train, Y_train,
                                epochs=epoch,
                                batch_size=batch_size,
                                validation_split=0.1,
                                callbacks=[model_checkpoint, stop],
                                verbose=1)
            train_end = time.time()
            
        print('=' * 10)
        print('***** evaluate *****')
        model.load_weights(file_conv)
        val_start = time.time()
        train_score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[0] // 48, verbose=0)
        val_end = time.time()
        
        print('Outflow:    RMSE    MAE | Inflow:    RMSE    MAE')
        print('Train score:',end=' ')
        np.set_printoptions(precision=6, suppress=True)
        print(np.array(train_score))
        test_start = time.time()
        test_score = model.evaluate(X_test,  Y_test, batch_size=Y_test.shape[0], verbose=0)
        test_end = time.time()
        print('Test  score:',end=' ')
        np.set_printoptions(precision=6, suppress=True)
        print(np.array(test_score))

        
        Pickup_RMSE[iterate_index, 0] = test_score[1]*MM/2
        Pickup_MAE[iterate_index, 0] = test_score[2]*MM/2
        Dropoff_RMSE[iterate_index, 0] = test_score[3]*MM/2
        Dropoff_MAE[iterate_index, 0] = test_score[4]*MM/2
        
        for_show=np.concatenate([Pickup_RMSE, Pickup_MAE, Dropoff_RMSE, Dropoff_MAE],axis=1)
        np.set_printoptions(precision=4, suppress=True)
        print('Outflow:    RMSE    MAE | Inflow:    RMSE    MAE')
        print(for_show)

    
        
        np.save('Exps/DeepSTN_eval_score.npy',[Pickup_RMSE, Pickup_MAE, Dropoff_RMSE, Dropoff_MAE])
            
        print(str(count)+'/'+str(count_sum))