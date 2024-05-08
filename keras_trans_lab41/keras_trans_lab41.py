# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 01:30:16 2021

@author: oscar
"""

import pandas as pd
import numpy as np


for bit in range(1):
    data_1 = pd.read_csv(f'D:\\py3710\\py_train_data\\0509yuan_16qam_17dB_cr10_TU6\\DNN_predict_ans\\0510_210_420_210_120_adadelta_epoch2000_batch100_bit0_preLLR.csv')
    #data_2 = pd.read_csv('C://Users//oscar//Desktop//experiments//1208lab2.5//rbf//predict_answer_lab2.5_rbfcsv.csv')
    test_nums = 30   ##28p4:1 28p8:2

    # def positive_and_negative_count(a):
    #     positive_count = (a > 0).sum()
    #     negative_count = (a < 0).sum()
    #     return print(f'正:\n{positive_count}',f'負:\n{negative_count}')



    character = "[]"
    #print(data_1.iloc[0,4])
    for i in range(0,421 * test_nums):#1684,1660
        for j in range(1,32):    
            data_1.iloc[i,j] = ''.join( x for x in data_1.iloc[i,j] if x not in character)
    print(data_1)
    data_1 = data_1.iloc[:,1:32]
    print(data_1)
    # data_3 = data_1.astype(float)
    # print('data3',data_3)
    # count = positive_and_negative_count(data_3)

    data2 = data_1*5
    # print(data2)
    aa = data_1.to_csv(f'D:\\py3710\\py_train_data\\0509yuan_16qam_17dB_cr10_TU6\\dnn_experiments\\0510_210_420_210_120_adadelta_epoch2000_batch100_bit0_preLLR.csv')
    # bb = data2.to_csv(f'D:\\py3710\\py_train_data\\0509yuan_16qam_17dB_cr10_TU6\\dnn_experiments\\0510_210_420_210_120_adam0.0001_epoch1500_batch100_bit0_preLLR_20.csv')
    #bb = data2.to_csv('D://MLforSchool//dnn_experiments//lab1_H_BER_16qam_10_15_100_DNN_senior.csv')#為了要算BER，估測出來的H用來等化
    #bb = data2.to_csv('C://Users//701//Desktop//check_BER//H_BER_95_DNN.csv')#為了要算BER，估測出來的H用來等化