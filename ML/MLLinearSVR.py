# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 19:29:29 2021

@author: oscar
"""
#linear演算法
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import *
import openpyxl
from openpyxl import Workbook
from pandas import DataFrame
import joblib
def transpose(list1):
    return[list(row) for row in zip(*list1)]

data_1 = pd.read_csv('F:\\Huang_ATSC\\MLforSchool\\data\\16qam_train\\train_3_15.csv')#drop 要刪除的行或列 要刪掉的話要加axis=1
x_train = data_1.drop(['ans','complex','id'],axis=1).values #feature1,feature2
print(x_train.shape)
data_3 = pd.read_csv('F:\\Huang_ATSC\\MLforSchool\\data\\16qam_test\\random_feature100_forTest.csv')
x_test = data_3.drop(['complex'],axis=1).values #feature1,feature2
for i in range(1,2):
    locals()['y_train'] = data_1['ans'].values
    #,loss='squared_epsilon_insensitive' 要做L2就要有這行(要加在c參數的後面) 做L1就把這行刪掉
    locals()['lin'] = LinearSVR(C=0.1,loss='squared_epsilon_insensitive',epsilon=0,tol=0.001,verbose=False)#C=0.1,epsilon=0, loss function 選L2 這組數據為linearsvr參數最佳化
    locals()['lin'].fit(x_train,locals()['y_train'])
#    joblib.dump(locals()['lin'], 'save//lin'+'.pkl')
    locals()['ans'] = locals()['lin'].predict(x_test)
    print(i)
#print(ans)
array = [ans]
a = transpose(array)
a = DataFrame(a,columns=['ans'])
b = a

#a.to_excel('predict_results_lab47-1_linearsvr.xlsx') 原本的 為了統一方便改成下一行
#a.to_excel('predict_answer_lab207_linearsvr.xlsx')
#a.to_csv('predict_answer_lab209-1_linearsvr.csv') 
a.to_csv('F:\\Huang_ATSC\\MLforSchool\\data\\16qam_test\\ans_for_test\\linearsvr_predict_answer_3_15_100.csv')
#b.to_csv('C://Users//701//Desktop//check_BER//H_BER_96_lin.csv')#為了要算BER，估測出來的H用來等化