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

from tools import change_i_to_j, change_all_positive, split_real_and_imag

first_nodes = 105
second_nodes =210
third_nodes = 105
four_nodes = 60
snr = 17
bit = 8 # The answer of b0 or b1 ...
i = 0
column = 100
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def transpose(list1):
    return[list(row) for row in zip(*list1)]

all_bit_mse_average = []
for i in range(0,1):
    
    mean_results = []
    mean_results_cal = []
    
    
    for j in range(10):
        train_feature_csv = pd.read_csv('D:\\MLforSchool\\data\\256qam_for_channel\\256qam_train\\lab4_256qamUi1_cr10_snr17_28884train_positive.csv')
        x_train_feature_i_to_j = train_feature_csv.iloc[0:, 1:].applymap(change_i_to_j)
        x_train_feature = x_train_feature_i_to_j.applymap(change_all_positive)
        list_x_train_complex =  x_train_feature.values.flatten()
        list_x_train_feature = list(map(split_real_and_imag, list_x_train_complex))
        train_ans_csv = pd.read_csv(f'D:\\MLforSchool\\data\\256qam_for_channel\\256qam_train\\ans\\lab4_LogMap_snr17_LLR_result_b{i}_28884.csv')
        y_train_ans = train_ans_csv.iloc[0: ,1:]
        list_y_train_ans = list(y_train_ans.values.flatten())

        test_ans_csv = pd.read_csv(f'D:\\MLforSchool\\data\\256qam_for_channel\\256qam_test\\ans\\lab3_LogMap_snr17_LLR_result_b{i}_6000.csv')
        test_feature_csv = pd.read_csv('D:\\MLforSchool\\data\\256qam_for_channel\\256qam_test\\lab3_256qamUi2_cr10_snr17_6000test.csv')
        test_feature_i_to_j = test_feature_csv.iloc[0:, 1:].applymap(change_i_to_j)
        test_feature = test_feature_i_to_j.applymap(change_all_positive)
        list_test_complex = test_feature.values.flatten()
        list_test_feature = list(map(split_real_and_imag, list_test_complex))


        dictionary_of_pridict_ans = {}
        predict_ans = []
        print("bit%d,第%d次"% (i,j+1))


        model = Sequential()#進行建造網路架構 在Sequential()裡面定義層數、激勵函數
        model.add(Dense(first_nodes, input_dim=2,  kernel_initializer='normal',activation='relu'))                               #加入神經層第一層(輸入14)輸出128 初始化器傳入 激活函數用relu #這邊的input一定要隨著特徵數量更改(pilot數乘2 因為實 虛 分開)
        model.add(Dense(second_nodes, input_dim=first_nodes,  kernel_initializer='normal',activation='relu'))
        model.add(Dense(third_nodes, input_dim=second_nodes,  kernel_initializer='normal',activation='relu'))
        model.add(Dense(four_nodes, input_dim=third_nodes,  kernel_initializer='normal',activation='relu'))
        model.add(Dense(1,  kernel_initializer='normal',activation='linear'))
        optimizer = Adam(decay=0.1)
        model.compile(loss='MSE', optimizer=optimizer, metrics = ['mse'])#設定model的loss和優化器(分別是MSE和adam) ,metrics=['mse','mape']
        epochs = 1000#代表疊帶40次(總共要用全部的訓練樣本重複跑幾回合)
        batch_size = 100#為你的输入指定一个固定的 batch 大小(每個iteration以100筆做計算)

        # history = model.fit(list_x_train_feature, list_y_train_ans, batch_size=batch_size, epochs=epochs ,verbose=1,validation_data=(list_x_valid_feature, list_y_valid_ans))
        model.fit(list_x_train_feature, list_y_train_ans, batch_size=batch_size, epochs=epochs ,verbose=1)
        #model.add(Dense(32, input_dim=x_train.shape[1],  kernel_initializer='normal',activation='relu'))
        # print("Saving model to disk \n")
        # mp = f"D://MLforSchool//DNN//lab4_snr{snr}_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}_LogMap_modelb{i}-2.h5"
        # model.save(mp)    #存model
        # summarize history for loss plt.plot(history.history['loss']) plt.plot(history.history['val_loss']) plt.title('model loss')
        # loss = history.history['loss']
        # val_loss = history.history['val_loss']

        # epochs = range(1, len(loss) + 1)
        # start_epoch = 10
        # plt.plot(epochs[start_epoch:],val_loss[start_epoch:], label = 'val_loss')
        # plt.plot(epochs[start_epoch:], loss[start_epoch:], label='loss')
        # plt.title('model loss')
        # plt.ylabel('MSE')
        # plt.xlabel('epoch')
        # plt.legend(['train_loss'], loc='upper right') 
        # plt.show()

        predict_ans= model.predict(list_test_feature) #訓練好model使用predict預測看看在訓練的model跑的回歸線
        flatten_predict_ans = predict_ans.flatten()
        index = 0
        for item in flatten_predict_ans:
            if index not in dictionary_of_pridict_ans:
                dictionary_of_pridict_ans[index] = []
            dictionary_of_pridict_ans[index].append(item)
            if len(dictionary_of_pridict_ans[index]) >= column:
                index = index + 1

        # print(dictionary_of_pridict_ans)

        csv = pd.DataFrame(dictionary_of_pridict_ans)

        # csv.T.to_csv(f'D://MLforSchool//dnn_experiments//channel//1211_256qam//snr17//mlp_lab1_256qam_snr17_10_15_LogMap_b{i}channel_{first_nodes}_{second_nodes}_{third_nodes}.csv')

        # predict_csv = pd.read_csv(f'D://MLforSchool//dnn_experiments//channel//1211_256qam//snr17//mlp_lab1_256qam_snr17_10_15_LogMap_b{i}channel_{first_nodes}_{second_nodes}_{third_nodes}.csv')
        
        csv.T.to_csv(f'D://MLforSchool//dnn_experiments//channel//test.csv')

        predict_csv = pd.read_csv(f'D://MLforSchool//dnn_experiments//channel//test.csv')
        def calc_mse(col):
            actual_values = test_ans_csv[col].values
            predicted_values = predict_csv[col].values
            mse = mean_squared_error(actual_values, predicted_values)
            # print("Mean Squared Error (MSE):", mse)
            return mse

        
        
        data_list = [str(i) for i in range(column)]
        calc_list = map(calc_mse,data_list)
        mean_result = statistics.mean(calc_list)
        print(j)
        print(mean_result)
        mean_results_cal.append(mean_result)
        
    
    a = sorted(mean_results_cal)
    print("Sorted Mean Results:", a)
    print("平均MSE:", statistics.mean(mean_results_cal))

# print("all_bit_mse_average",all_bit_mse_average)
