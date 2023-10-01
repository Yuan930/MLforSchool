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


from flatten import *


bit = 2 # The answer of b0 or b1 ...


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def transpose(list1):
    return[list(row) for row in zip(*list1)]


def split_real_and_imag(comp):
    array = comp.replace('j', '').split('+')
    return [float(array[0]), float(array[1])]

train_feature_csv = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_channel\\16qam_train\\fU_for_train_positive.csv')
x_train_feature = train_feature_csv.iloc[0:, 1:]
list_x_train_complex =  flatten(x_train_feature.values)
list_x_train_feature = list(map(split_real_and_imag, list_x_train_complex))
train_ans_csv = pd.read_csv(f'D:\\MLforSchool\\data\\16qam_for_channel\\16qam_train\\ans\\result_b{bit}.csv')
y_train_ans = train_ans_csv.iloc[0: ,1:]
list_y_train_ans = flatten(y_train_ans.values)

valid_feature_csv = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_channel\\16qam_valid\\fU_for_valid_positive.csv')
x_valid_feature = valid_feature_csv.iloc[0:, 1:]
list_x_valid_complex =  flatten(x_valid_feature.values)
list_x_valid_feature = list(map(split_real_and_imag, list_x_valid_complex))
valid_ans_csv = pd.read_csv(f'D:\\MLforSchool\\data\\16qam_for_channel\\16qam_valid\\ans\\result_b{bit}.csv')
y_valid_ans = valid_ans_csv.iloc[0: ,1:]
list_y_valid_ans = flatten(y_valid_ans.values)

test_feature_csv = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_channel\\16qam_test\\fU_for_test_positive.csv')
test_feature = test_feature_csv.iloc[0:, 1:]
list_test_complex =  flatten(test_feature.values)
list_test_feature = list(map(split_real_and_imag, list_test_complex))


dictionary_of_pridict_ans = {}
predict_ans = []


model = Sequential()#進行建造網路架構 在Sequential()裡面定義層數、激勵函數
model.add(Dense(35, input_dim=2,  kernel_initializer='normal',activation='relu'))                               #加入神經層第一層(輸入14)輸出128 初始化器傳入 激活函數用relu #這邊的input一定要隨著特徵數量更改(pilot數乘2 因為實 虛 分開)
model.add(Dense(70, input_dim=35,  kernel_initializer='normal',activation='relu'))
model.add(Dense(35, input_dim=70,  kernel_initializer='normal',activation='relu'))
model.add(Dense(20, input_dim=35,  kernel_initializer='normal',activation='relu'))
model.add(Dense(1,  kernel_initializer='normal',activation='linear'))
model.compile(loss='MSE', optimizer='adam')#設定model的loss和優化器(分別是MSE和adam) ,metrics=['mse','mape']
epochs = 40#代表疊帶40次(總共要用全部的訓練樣本重複跑幾回合)
batch_size = 100#為你的输入指定一个固定的 batch 大小(每個iteration以100筆做計算)

model.fit(list_x_train_feature, list_y_train_ans, batch_size=100, epochs=epochs ,verbose=1,validation_data=(list_x_valid_feature, list_y_valid_ans))
#model.add(Dense(32, input_dim=x_train.shape[1],  kernel_initializer='normal',activation='relu'))


predict_ans= model.predict(list_test_feature) #訓練好model使用predict預測看看在訓練的model跑的回歸線
flatten_predict_ans = flatten(predict_ans)
index = 0
for item in flatten_predict_ans:
    if index not in dictionary_of_pridict_ans:
        dictionary_of_pridict_ans[index] = []
    dictionary_of_pridict_ans[index].append(item)
    if len(dictionary_of_pridict_ans[index]) >= 10:
        index = index + 1

print(dictionary_of_pridict_ans)

csv = pd.DataFrame(dictionary_of_pridict_ans)

csv.T.to_csv('D://MLforSchool//dnn_experiments//channel//mlp_predict_answer_lab1_16qam_10_15_100_channel_senior.csv')
# for key in a.key():
# arr = [a0,a1,a2,a3]


# print(arr)

# print(arr)
# print(arr[0][1])
# print(arr[1][1])
# print(arr[0][1]-arr[1][1])
# a = transpose(arr)
# a = DataFrame(a)

# a.columns = [f'b{i}' for i in range(4)]
# a.to_csv('D://MLforSchool//dnn_experiments//channel//mlp_predict_answer_lab1_16qam_10_15_100_channel_senior.csv')




# a = []
# dictionary_of_pridict_ans = {}
# a = [[1],[2],[3],[4],[5]]
# flatten_a = flatten(a)
# dic = {}
# # print(len(a))
# index = 0
# for item in flatten_a:
#     if index not in dic:
#         dic[index] = []
#     dic[index].append(item)
#     if len(dic[index]) >= 2:
#         index = index + 1

# print(dic)

# for key in dic.keys():
#     result = {}
#     index = 0
#     for item in dic[key]:
#         if index not in result:
#             result[index] = []
#         result[index].append(item)
#         if (len(result[index]) >= 2):  #根據測試資料的列數更改
#             index = index + 1
#     print(result)


