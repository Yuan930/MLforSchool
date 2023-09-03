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
data_1 = pd.read_csv('D://py3710//py_train_data//TU6 16dB train TU6 16dB test (pilot4+3)//train1_28.csv')
x_train = data_1.drop(['ans62','ans61','ans60','ans59','ans58','ans57','ans56','ans55','ans54','ans53','ans52','ans51','ans50',
                       'ans49','ans48','ans47','ans46','ans45','ans44','ans43','ans42','ans41','ans40','ans39','ans38','ans37',
                       'ans36','ans35','ans34','ans33','ans32','ans31','ans30','ans29','ans28','ans27','ans26','ans25','ans24',
                       'ans23','ans22','ans21','ans20','ans19','ans18','ans17','ans16','ans15','ans14','ans13','ans12','ans11',
                       'ans10','ans9','ans8','ans7','ans6','ans5','ans4','ans3','ans2','ans1','id'],axis=1).values

data_2 = pd.read_csv('D://py3710//py_train_data//TU6 16dB train TU6 16dB test (pilot4+3)//valid1.csv')
x_valid = data_2.drop(['ans62','ans61','ans60','ans59','ans58','ans57','ans56','ans55','ans54','ans53','ans52','ans51','ans50',
                       'ans49','ans48','ans47','ans46','ans45','ans44','ans43','ans42','ans41','ans40','ans39','ans38','ans37',
                       'ans36','ans35','ans34','ans33','ans32','ans31','ans30','ans29','ans28','ans27','ans26','ans25','ans24',
                       'ans23','ans22','ans21','ans20','ans19','ans18','ans17','ans16','ans15','ans14','ans13','ans12','ans11',
                       'ans10','ans9','ans8','ans7','ans6','ans5','ans4','ans3','ans2','ans1','id'],axis=1).values
data_3 = pd.read_csv('D://py3710//py_train_data//TU6 16dB train TU6 16dB test (pilot4+3)//test1_28p8.csv')
x_test = data_3.drop(['id'],axis=1).values




for i in range(1,63):
    locals()['y_train'+str(i)] = data_1['ans'+str(i)].values

#print(y_train3)
    locals()['y_valid'+str(i)] = data_2['ans'+str(i)].values
    model = Sequential()#進行建造網路架構 在Sequential()裡面定義層數、激勵函數
    model.add(Dense(35, input_dim=14,  kernel_initializer='normal',activation='relu'))#加入神經層第一層(輸入14)輸出128 初始化器傳入 激活函數用relu #這邊的input一定要隨著特徵數量更改(pilot數乘2 因為實 虛 分開)
    model.add(Dense(70, input_dim=35,  kernel_initializer='normal',activation='relu'))
    model.add(Dense(35, input_dim=70,  kernel_initializer='normal',activation='relu'))
    model.add(Dense(20, input_dim=35,  kernel_initializer='normal',activation='relu'))
    model.add(Dense(1,  kernel_initializer='normal',activation='linear'))
    model.compile(loss='MSE', optimizer='adam')#設定model的loss和優化器(分別是MSE和adam) ,metrics=['mse','mape']
    epochs = 40#代表疊帶40次(總共要用全部的訓練樣本重複跑幾回合)
    batch_size = 100#為你的输入指定一个固定的 batch 大小(每個iteration以100筆做計算)

    model.fit(x_train, locals()['y_train'+str(i)], batch_size=100, epochs=epochs ,verbose=1,validation_data=(x_valid, locals()['y_valid'+str(i)]))
#model.add(Dense(32, input_dim=x_train.shape[1],  kernel_initializer='normal',activation='relu'))


    locals()['anss'+str(i)] = model.predict(x_test) #訓練好model使用predict預測看看在訓練的model跑的回歸線
    print(i)
arr = [anss1,anss2,anss3,anss4,anss5,anss6,anss7,anss8,anss9,anss10,anss11,anss12,anss13,anss14,anss15,anss16,anss17,anss18,anss19,
         anss20,anss21,anss22,anss23,anss24,anss25,anss26,anss27,anss28,anss29,anss30,anss31,anss32,anss33,anss34,anss35,anss36,
         anss37,anss38,anss39,anss40,anss41,anss42,anss43,anss44,anss45,anss46,anss47,anss48,anss49,anss50,anss51,anss52,anss53,
         anss54,anss55,anss56,anss57,anss58,anss59,anss60,anss61,anss62]
print(arr)

print(arr)
print(arr[0][1])
print(arr[1][1])
print(arr[0][1]-arr[1][1])
a = transpose(arr)
a = DataFrame(a)

#answer = pd.read_csv('C://Users//oscar//Desktop//spyder//1123_6pilot_31ans//answer.csv')
#a.to_excel('mlp_predict_answer_lab52_epoch50.xlsx')
a.to_csv('D://py3710//dnn_experiments//mlp_predict_answer_lab1_28p8_epoch40.csv')
#fn=str(epochs)+'_1'+str(batch_size)

#model.save('C://Users//oscar//Desktop//spyder//keras'+fn+'.h5')
