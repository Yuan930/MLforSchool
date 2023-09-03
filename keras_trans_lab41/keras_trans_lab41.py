# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 01:30:16 2021

@author: oscar
"""

import pandas as pd
import numpy as np

data_1 = pd.read_csv('D://py3710////dnn_experiments//mlp_predict_answer_lab1_28p8_epoch40.csv')
#data_2 = pd.read_csv('C://Users//oscar//Desktop//experiments//1208lab2.5//rbf//predict_answer_lab2.5_rbfcsv.csv')
test_nums = 2   ##28p4:1 28p8:2

print(data_1.iloc[0,1])
print(data_1)
string = "Hey! What's up?"
characters = "'!?"

character = "[]"
print(data_1.iloc[0,62])
for i in range(0,1684*test_nums):#1684
    for j in range(1,63):
        
        data_1.iloc[i,j] = ''.join( x for x in data_1.iloc[i,j] if x not in character)
print(data_1)
data_1 = data_1.iloc[:,1:63]
print(data_1)
data2 = data_1
aa = data_1.to_csv('D://py3710//py_train_data//TU6 16dB train TU6 16dB test (pilot4+3)//dnn_experiments//trans_answer_mlp_lab1_28p8_epoch40.csv')
#bb = data2.to_csv('C://Users//701//Desktop//check_BER//H_BER_95_DNN.csv')#為了要算BER，估測出來的H用來等化