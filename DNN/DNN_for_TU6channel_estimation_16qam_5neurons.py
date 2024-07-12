# -*- coding: utf-8 -*-
"""
Created on Sun Oct 1 20:12:12 2023

@author: Ting Yuan Huang
"""
from sklearn.metrics import mean_squared_error
import statistics
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import *
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import DataFrame
import os
import re
import tensorflow as tf
from keras import backend as K
from tools import change_i_to_j, change_all_positive, remove_parentheses, split_real_and_imag

first_nodes = 126   #105*4/5
second_nodes = 252  #210*4/5
third_nodes = 189   #增加的隱藏層
fourth_nodes = 126   #105*4/5
fifth_nodes = 72    #60*4/5

column = 31
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def transpose(list1):
    return[list(row) for row in zip(*list1)]
def feature_csv_change_to_list(feature_csv):
    feature_i_to_j = np.vectorize(change_i_to_j)(feature_csv)
    # feature = np.vectorize(change_all_positive)(feature_i_to_j)
    list_complex = feature_i_to_j.flatten()

    return split_real_and_imag(list_complex)

def complex_to_positive(complex_str):
    real_imag_parts = complex_str.replace('i', 'j').replace('+', ' ').split()

    if 'j' not in real_imag_parts[0]:
        real_part = float(real_imag_parts[0]) if real_imag_parts[0] else 0.0
        imag_part = 0.0
    else:
        real_part, imag_part = map(float, real_imag_parts[0].split('j'))

    positive_real = abs(real_part)
    positive_imag = abs(imag_part)

    return positive_real, positive_imag

def train_data(bit, j):
    train_Ui_csv = pd.read_csv(f'D:\\py3710\\py_train_data\\0509yuan_16qam_12dB_cr10_TU6\\train_TU6_16qam_Y_BER_8p7_12dB_70.csv')
    Ui_train = train_Ui_csv[str(j)].values
    Ui_train_real_imag = np.array(feature_csv_change_to_list(Ui_train))
    pilot_data_csv = pd.read_csv(f'D:\\py3710\\py_train_data\\0509yuan_16qam_12dB_cr10_TU6\\train_pilot_8p7_TU6_16qam_12dB_70.csv')
    pilot_train = pilot_data_csv.drop(['id'],axis = 1).values
    var_train = np.full((Ui_train_real_imag.shape[0], 1), 0.0631)
    train_ans_csv = pd.read_csv(f'D:\\py3710\\py_train_data\\0509yuan_16qam_12dB_cr10_TU6\\t70_train_data_LogMapllr_b{bit}.csv')
    train_ans_csv = train_ans_csv/5
    train_ans = train_ans_csv[str(j)].values
    return Ui_train_real_imag, pilot_train, var_train, train_ans

def test_data(j):
    test_Ui_csv = pd.read_csv(f'D:\\py3710\\py_train_data\\0509yuan_16qam_12dB_cr10_TU6\\test_TU6_16qam_Y_BER_8p7_12dB_30.csv')
    Ui_test = test_Ui_csv[str(j)].values
    Ui_test_real_imag = np.array(feature_csv_change_to_list(Ui_test))
    
    test_predictH_data_csv = pd.read_csv(f'D:\\py3710\\py_train_data\\0509yuan_16qam_12dB_cr10_TU6\\test_pilot_8p7_TU6_16qam_12dB_30.csv')
    pilot_test = test_predictH_data_csv.drop(['id'],axis = 1).values
    var_test = np.full((Ui_test_real_imag.shape[0], 1), 0.0631)
    return Ui_test_real_imag, pilot_test,var_test

for i in range(4):
    
    mean_results = []
    mean_results_cal = []
    for j in range(1,32):
        ########################################TRAIN#######################################
        Ui_train_real_imag, pilot_train, var_train, train_ans = train_data(i, j)
        # print('Ui_train_real_imag',Ui_train_real_imag)
        # print('pilot_train',pilot_train)
        # print('train_ans',train_ans)
        train_feature = np.hstack((Ui_train_real_imag, pilot_train,var_train))
        
#         #########################################TEST#####################################
        # Ui_test_real_imag, pilot_test,var_test = test_data(j)
        # test_ans_csv = pd.read_csv(f'D:\\py3710\\py_train_data\\0425yuan_16qam_8dB_cr10_TU6\\test_data_LogMapllr_b{i}.csv')
        # test_ans = test_ans_csv.drop(['id'],axis =1).values

        # test_feature = np.hstack((Ui_test_real_imag, pilot_test,var_test))
#         dictionary_of_pridict_ans = {}
#         predict_ans = []
#         print("bit%d,第%d次"% (i,j))



        model = Sequential()#進行建造網路架構 在Sequential()裡面定義層數、激勵函數
        model.add(Dense(first_nodes, input_dim=33,  kernel_initializer='normal',activation='relu'))                               #加入神經層第一層(輸入14)輸出128 初始化器傳入 激活函數用relu #這邊的input一定要隨著特徵數量更改(pilot數乘2 因為實 虛 分開)
        model.add(Dense(second_nodes, input_dim=first_nodes,  kernel_initializer='normal',activation='relu'))
        model.add(Dense(third_nodes, input_dim=second_nodes,  kernel_initializer='normal',activation='relu'))
        model.add(Dense(fourth_nodes, input_dim=third_nodes,  kernel_initializer='normal',activation='relu'))
        model.add(Dense(fifth_nodes, input_dim=fourth_nodes,  kernel_initializer='normal',activation='relu'))
        model.add(Dense(1,  kernel_initializer='normal',activation='linear'))
        # optimizer = Adam(learning_rate=0.0001)
        model.compile(loss='MSE', optimizer='adadelta', metrics = ['mse'])#設定model的loss和優化器(分別是MSE和adam) ,metrics=['mse','mape']
        epochs = 1000#代表疊帶40次(總共要用全部的訓練樣本重複跑幾回合)
        batch_size = 100#為你的输入指定一个固定的 batch 大小(每個iteration以100筆做計算)

        history = model.fit(train_feature, train_ans, batch_size=batch_size, epochs=epochs ,verbose=1)
        # model.add(Dense(32, input_dim=x_train.shape[1],  kernel_initializer='normal',activation='relu'))
        print("Saving model to disk \n")
        mp = f"D:\\py3710\\py_train_data\\0509yuan_16qam_12dB_cr10_TU6\\DNN_hdf5\\0712_train70_linear(-20,20)_{first_nodes}_{second_nodes}_{third_nodes}_{fourth_nodes}_{fifth_nodes}adadelta_epoch{epochs}_batch{batch_size}_bit{i}_preLLR{j}.h5"
        model.save(mp)    #存model
        
        ##########################畫圖###################################
        loss = history.history['loss']

        epochs = range(1, len(loss) + 1)
        start_epoch = 0
        plt.figure()

        plt.yscale('log')  # Set y-axis to logarithmic scale
        plt.plot(epochs[start_epoch:], loss[start_epoch:], label='loss')
        plt.title('model loss')
        plt.ylabel('MSE')
        plt.xlabel('epoch')
        plt.ylim(10**-1, 10**1)
        plt.legend(['train_loss'], loc='upper right') 
        save_path = f'D:\\py3710\\py_train_data\\0509yuan_16qam_12dB_cr10_TU6\\DNN_pic\\0712_train70_linear(-20,20)_{first_nodes}_{second_nodes}_{third_nodes}_{fourth_nodes}_{fifth_nodes}adadelta_epoch{epochs}_batch{batch_size}_bit{i}_hdf5LLR{j}.png'
        plt.savefig(save_path)
        print(j)
        plt.close()
        # plt.show()
#         #######################################################################################################
    #     locals()['anss'+str(j)] = model.predict(test_feature) #訓練好model使用predict預測看看在訓練的model跑的回歸線
    #     locals()['anss'+str(j)] = np.clip(locals()['anss'+str(j)], -20, 20)
    #     print(i,'column',j)
    # arr = [anss1,anss2,anss3,anss4,anss5,anss6,anss7,anss8,anss9,anss10,anss11,anss12,anss13,anss14,anss15,anss16,anss17,anss18,anss19,anss20,anss21,anss22,anss23,anss24,anss25,anss26,anss27,anss28,anss29,anss30,anss31]
    # print(arr)

    # print(arr)
    # print(arr[0][1])
    # print(arr[1][1])
    # print(arr[0][1]-arr[1][1])
    # a = transpose(arr)
    # a = DataFrame(a)
    # a.to_csv(f'D:\\py3710\\py_train_data\\0509yuan_16qam_12dB_cr10_TU6\\DNN_predict_ans\\0523_(-20,20)_{first_nodes}_{second_nodes}_{third_nodes}_{fourth_nodes}_{fifth_nodes}adadelta_epoch{epochs}_batch{batch_size}_bit{i}_preLLR.csv')
