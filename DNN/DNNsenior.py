# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 20:12:12 2021

@author: oscar
"""

from keras.models import Sequential
from keras.layers import *
from keras.callbacks import *
import pandas as pd
import numpy as np
from pandas import DataFrame
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def transpose(list1):
    return[list(row) for row in zip(*list1)]
data_1 = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_randomfeature\\16qam_train\\ans\\train_10_15_1000.csv')
x_train = data_1.drop(['b0','b1','b2','b3','id'],axis=1).values

data_2 = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_randomfeature\\16qam_valid\\ans\\valid_10_15_100.csv')
x_valid = data_2.drop(['b0','b1','b2','b3','id'],axis=1).values
data_3 = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_randomfeature\\16qam_test\\random_feature100_forTest.csv')
x_test = data_3.drop(['complex'],axis=1).values




for i in range(0,4):
    locals()['y_train'+str(i)] = data_1['b'+str(i)].values

#print(y_train3)
    locals()['y_valid'+str(i)] = data_2['b'+str(i)].values
    model = Sequential()#進行建造網路架構 在Sequential()裡面定義層數、激勵函數
    model.add(Dense(35, input_dim=2,  kernel_initializer='normal',activation='relu'))                               #加入神經層第一層(輸入14)輸出128 初始化器傳入 激活函數用relu #這邊的input一定要隨著特徵數量更改(pilot數乘2 因為實 虛 分開)
    model.add(Dense(70, input_dim=35,  kernel_initializer='normal',activation='relu'))
    model.add(Dense(35, input_dim=70,  kernel_initializer='normal',activation='relu'))
    model.add(Dense(20, input_dim=35,  kernel_initializer='normal',activation='relu'))
    model.add(Dense(1,  kernel_initializer='normal',activation='linear'))
    model.compile(loss='MSE', optimizer='adam')#設定model的loss和優化器(分別是MSE和adam) ,metrics=['mse','mape']
    epochs = 40#代表疊帶40次(總共要用全部的訓練樣本重複跑幾回合)
    batch_size = 100#為你的输入指定一个固定的 batch 大小(每個iteration以100筆做計算)

    model.fit(x_train, locals()['y_train'+str(i)], batch_size=100, epochs=epochs ,verbose=1,validation_data=(x_valid, locals()['y_valid'+str(i)]))
#model.add(Dense(32, input_dim=x_train.shape[1],  kernel_initializer='normal',activation='relu'))


    locals()['bb'+str(i)] = model.predict(x_test) #訓練好model使用predict預測看看在訓練的model跑的回歸線
    print(i)
arr = [bb0,bb1,bb2,bb3]
print(arr)

print(arr)
print(arr[0][1])
print(arr[1][1])
print(arr[0][1]-arr[1][1])
a = transpose(arr)
a = DataFrame(a)

#answer = pd.read_csv('C://Users//oscar//Desktop//spyder//1123_6pilot_31ans//answer.csv')
#a.to_excel('mlp_predict_answer_lab52_epoch50.xlsx')
a.columns = [f'b{i}' for i in range(4)]
a.to_csv('D://MLforSchool//dnn_experiments//mlp_predict_answer_lab1_16qam_10_15_100_neurons15.csv')
#fn=str(epochs)+'_1'+str(batch_size)

#model.save('C://Users//oscar//Desktop//spyder//keras'+fn+'.h5')
