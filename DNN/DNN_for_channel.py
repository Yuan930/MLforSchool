# -*- coding: utf-8 -*-
"""
Created on Sun Oct 1 20:12:12 2023

@author: Ting Yuan Huang
"""

from keras.models import Sequential
from keras.layers import *
from keras.callbacks import *
import pandas as pd
import numpy as np
from pandas import DataFrame
import os
import re

from tools import change_i_to_j, change_all_positive, split_real_and_imag


bit = 3 # The answer of b0 or b1 ...
column = 100
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def transpose(list1):
    return[list(row) for row in zip(*list1)]
for i in range(bit+1):
    train_feature_csv = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_channel\\16qam_train\\lab4_16qamUi_coderate10_snr8_train.csv')
    x_train_feature_i_to_j = train_feature_csv.iloc[0:, 1:].applymap(change_i_to_j)
    x_train_feature = x_train_feature_i_to_j.applymap(change_all_positive)
    list_x_train_complex =  x_train_feature.values.flatten()
    list_x_train_feature = list(map(split_real_and_imag, list_x_train_complex))
    train_ans_csv = pd.read_csv(f'D:\\MLforSchool\\data\\16qam_for_channel\\16qam_train\\ans\\lab4_maximum_LLR_result_b{i}.csv')
    y_train_ans = train_ans_csv.iloc[0: ,1:]
    list_y_train_ans = list(y_train_ans.values.flatten())

    valid_feature_csv = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_channel\\16qam_valid\\lab4_16qamUi_coderate10_snr8_valid.csv')
    x_valid_feature_i_to_j = valid_feature_csv.iloc[0:, 1:].applymap(change_i_to_j)
    x_valid_feature = x_valid_feature_i_to_j.applymap(change_all_positive)
    list_x_valid_complex = x_valid_feature.values.flatten()
    list_x_valid_feature = list(map(split_real_and_imag, list_x_valid_complex))
    valid_ans_csv = pd.read_csv(f'D:\\MLforSchool\\data\\16qam_for_channel\\16qam_valid\\ans\\lab4_maximum_LLR_result_b{i}.csv')
    y_valid_ans = valid_ans_csv.iloc[0: ,1:]
    list_y_valid_ans = list(y_valid_ans.values.flatten())

    test_feature_csv = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_channel\\16qam_test\\lab4_16qamUi_coderate10_snr8_test.csv')
    test_feature_i_to_j = test_feature_csv.iloc[0:, 1:].applymap(change_i_to_j)
    test_feature = test_feature_i_to_j.applymap(change_all_positive)
    list_test_complex = test_feature.values.flatten()
    list_test_feature = list(map(split_real_and_imag, list_test_complex))


    dictionary_of_pridict_ans = {}
    predict_ans = []
    print(i)
    print(list_y_train_ans[0], list_y_train_ans[151])


    # model = Sequential()#進行建造網路架構 在Sequential()裡面定義層數、激勵函數
    # model.add(Dense(70, input_dim=2,  kernel_initializer='normal',activation='relu'))                               #加入神經層第一層(輸入14)輸出128 初始化器傳入 激活函數用relu #這邊的input一定要隨著特徵數量更改(pilot數乘2 因為實 虛 分開)
    # model.add(Dense(140, input_dim=70,  kernel_initializer='normal',activation='relu'))
    # model.add(Dense(70, input_dim=140,  kernel_initializer='normal',activation='relu'))
    # model.add(Dense(40, input_dim=70,  kernel_initializer='normal',activation='relu'))
    # model.add(Dense(1,  kernel_initializer='normal',activation='linear'))
    # model.compile(loss='MSE', optimizer='adam')#設定model的loss和優化器(分別是MSE和adam) ,metrics=['mse','mape']
    # epochs = 40#代表疊帶40次(總共要用全部的訓練樣本重複跑幾回合)
    # batch_size = 100#為你的输入指定一个固定的 batch 大小(每個iteration以100筆做計算)

    # model.fit(list_x_train_feature, list_y_train_ans, batch_size=100, epochs=epochs ,verbose=1,validation_data=(list_x_valid_feature, list_y_valid_ans))
    # #model.add(Dense(32, input_dim=x_train.shape[1],  kernel_initializer='normal',activation='relu'))


    # predict_ans= model.predict(list_test_feature) #訓練好model使用predict預測看看在訓練的model跑的回歸線
    # flatten_predict_ans = predict_ans.flatten()
    # index = 0
    # for item in flatten_predict_ans:
    #     if index not in dictionary_of_pridict_ans:
    #         dictionary_of_pridict_ans[index] = []
    #     dictionary_of_pridict_ans[index].append(item)
    #     if len(dictionary_of_pridict_ans[index]) >= column:
    #         index = index + 1

    # # print(dictionary_of_pridict_ans)

    # csv = pd.DataFrame(dictionary_of_pridict_ans)

    # csv.T.to_csv(f'D://MLforSchool//dnn_experiments//channel//mlp_answer_lab4_16qam_10_15_b{i}channel_70_140_70_40.csv')


