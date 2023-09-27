# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 01:30:16 2021

@author: oscar
"""

import pandas as pd
import numpy as np

data_1 = pd.read_csv('D://MLforSchool//dnn_experiments//independent_ans//mlp_predict_answer_lab2_16qam_10_15_100_1.4_senior_b2.csv')
#data_2 = pd.read_csv('C://Users//oscar//Desktop//experiments//1208lab2.5//rbf//predict_answer_lab2.5_rbfcsv.csv')
test_nums = 2   ##28p4:1 28p8:2

print(data_1.iloc[0,1])
print(data_1)
string = "Hey! What's up?"
characters = "'!?"

character = "[]"
#print(data_1.iloc[0,4])
for i in range(0,100):#1684
    #for j in range(1,5):
    for j in range(1,2):    
        data_1.iloc[i,j] = ''.join( x for x in data_1.iloc[i,j] if x not in character)
print(data_1)
data_1 = data_1.iloc[:,1]
print(data_1)
data_3 = data_1.astype(float)
positive_count = (data_3 > 0).sum()
negative_count = (data_3 < 0).sum()
print(f'正: {positive_count}')
print(f'負: {negative_count}')

data2 = data_1
# aa = data_1.to_csv('D://MLforSchool//dnn_experiments//independent_ans//lab2_trans_answer_mlp_16qam_10_15_100_1.4_neurous15_b2.csv')
#bb = data2.to_csv('D://MLforSchool//dnn_experiments//lab1_H_BER_16qam_10_15_100_DNN_senior.csv')#為了要算BER，估測出來的H用來等化
#bb = data2.to_csv('C://Users//701//Desktop//check_BER//H_BER_95_DNN.csv')#為了要算BER，估測出來的H用來等化