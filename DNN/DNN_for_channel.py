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
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import DataFrame
import os
import re

from tools import change_i_to_j, change_all_positive, split_real_and_imag

first_nodes = 140
second_nodes =280
third_nodes = 140
four_nodes = 80
bit = 4 # The answer of b0 or b1 ...
i = 3
column = 100
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def transpose(list1):
    return[list(row) for row in zip(*list1)]

train_feature_csv = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_channel\\16qam_train\\lab4_16qamUi_coderate10_snr8_train.csv')
x_train_feature_i_to_j = train_feature_csv.iloc[0:, 1:].applymap(change_i_to_j)
x_train_feature = x_train_feature_i_to_j.applymap(change_all_positive)
list_x_train_complex =  x_train_feature.values.flatten()
list_x_train_feature = list(map(split_real_and_imag, list_x_train_complex))
train_ans_csv = pd.read_csv(f'D:\\MLforSchool\\data\\16qam_for_channel\\16qam_train\\ans\\lab4_Max_Log_LLR_result_b{i}.csv')
y_train_ans = train_ans_csv.iloc[0: ,1:]
list_y_train_ans = list(y_train_ans.values.flatten())

test_ans_csv = pd.read_csv(f'D:\\MLforSchool\\data\\16qam_for_channel\\16qam_test\\ans\\lab4_Max_Log_LLR_result_b{i}.csv')


# valid_feature_csv = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_channel\\16qam_valid\\lab4_16qamUi_coderate10_snr8_valid.csv')
# x_valid_feature_i_to_j = valid_feature_csv.iloc[0:, 1:].applymap(change_i_to_j)
# x_valid_feature = x_valid_feature_i_to_j.applymap(change_all_positive)
# list_x_valid_complex = x_valid_feature.values.flatten()
# list_x_valid_feature = list(map(split_real_and_imag, list_x_valid_complex))
# valid_ans_csv = pd.read_csv(f'D:\\MLforSchool\\data\\16qam_for_channel\\16qam_valid\\ans\\lab4_Max_Log_LLR_result_b{i}.csv')
# y_valid_ans = valid_ans_csv.iloc[0: ,1:]
# list_y_valid_ans = list(y_valid_ans.values.flatten())

test_feature_csv = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_channel\\16qam_test\\lab4_16qamUi_coderate10_snr8_test.csv')
test_feature_i_to_j = test_feature_csv.iloc[0:, 1:].applymap(change_i_to_j)
test_feature = test_feature_i_to_j.applymap(change_all_positive)
list_test_complex = test_feature.values.flatten()
list_test_feature = list(map(split_real_and_imag, list_test_complex))


dictionary_of_pridict_ans = {}
predict_ans = []
print(i)


model = Sequential()#進行建造網路架構 在Sequential()裡面定義層數、激勵函數
model.add(Dense(first_nodes, input_dim=2,  kernel_initializer='normal',activation='relu'))                               #加入神經層第一層(輸入14)輸出128 初始化器傳入 激活函數用relu #這邊的input一定要隨著特徵數量更改(pilot數乘2 因為實 虛 分開)
model.add(Dense(second_nodes, input_dim=first_nodes,  kernel_initializer='normal',activation='relu'))
model.add(Dense(third_nodes, input_dim=second_nodes,  kernel_initializer='normal',activation='relu'))
model.add(Dense(four_nodes, input_dim=third_nodes,  kernel_initializer='normal',activation='relu'))
model.add(Dense(1,  kernel_initializer='normal',activation='linear'))
model.compile(loss='MSE', optimizer='adam', metrics = ['mse'])#設定model的loss和優化器(分別是MSE和adam) ,metrics=['mse','mape']
epochs = 40#代表疊帶40次(總共要用全部的訓練樣本重複跑幾回合)
batch_size = 100#為你的输入指定一个固定的 batch 大小(每個iteration以100筆做計算)

# history = model.fit(list_x_train_feature, list_y_train_ans, batch_size=batch_size, epochs=epochs ,verbose=1,validation_data=(list_x_valid_feature, list_y_valid_ans))
history =  model.fit(list_x_train_feature, list_y_train_ans, batch_size=batch_size, epochs=epochs ,verbose=1)
#model.add(Dense(32, input_dim=x_train.shape[1],  kernel_initializer='normal',activation='relu'))
print("Saving model to disk \n")
mp = "D://MLforSchool//DNN//iris_modelb3.h5"
model.save(mp)
# summarize history for loss plt.plot(history.history['loss']) plt.plot(history.history['val_loss']) plt.title('model loss')
loss = history.history['loss']
# val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)
start_epoch = 0
# plt.plot(epochs[start_epoch:],val_loss[start_epoch:], label = 'val_loss')
plt.plot(epochs[start_epoch:], loss[start_epoch:], label='loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss'], loc='upper right') 
plt.show()

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

csv.T.to_csv(f'D://MLforSchool//dnn_experiments//channel//mlp_lab5_16qam_10_15_Max_Log_b{i}channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')



predict_csv = pd.read_csv(f'D://MLforSchool//dnn_experiments//channel//mlp_lab5_16qam_10_15_Max_Log_b{i}channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')
def calc_mse(col):
    actual_values = test_ans_csv[col].values
    predicted_values = predict_csv[col].values
    mse = mean_squared_error(actual_values, predicted_values)
    # print("Mean Squared Error (MSE):", mse)
    return mse

n = 100
data_list = [str(i) for i in range(n)]
calc_list = map(calc_mse,data_list)
print(statistics.mean(calc_list))

