# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 20:14:57 2021

@author: oscar
"""

import pandas as pd
from pandas import DataFrame
import numpy as np
#predict = pd.read_csv('C://Users//701//Desktop//experiments//1212lab208//predict_answer_lab208_rbf.csv') #linearsvr or rbf
predict = pd.read_csv('D://py3710//py_train_data//TU6 16dB train TU6 16dB test_clipping (pilot4+3)//retest//predict_answer_lab3_linearsvr_28p16.csv') #linearsvr or rbf
print(predict)

print(predict.iloc[0,2])
print(predict.iloc[0,3])
print(predict.iloc[1,0])
print(predict.iloc[2,0])
print(predict.iloc[2,1])
def transpose(list1):
    return[list(row) for row in zip(*list1)]



answer = pd.read_csv('D://py3710//py_train_data//TU6 16dB train TU6 16dB test_clipping (pilot4+3)//retest//answer3_28p16.csv')
print(answer)
print(answer.iloc[0,1])

print(answer.iloc[0,2])
print(answer.iloc[0,3])
print(answer.iloc[1,0]) 
print(answer.iloc[2,0])
print(answer.iloc[2,1])
print(predict.iloc[0,32])
print(answer.iloc[0,32])
print((predict.iloc[0,1]-answer.iloc[0,1])**2+(predict.iloc[0,32]-answer.iloc[0,32])**2)
print(predict.iloc[99,1])
#第二個pilot後面一位的頻率響應square error即第一筆的第一個答案之square error
#預測的第一個剪掉答案的第一個+預測需部第一個減答案虛不第一個
#(ans1-pred1) + (ans32-pred32)
#a1 = (predict.iloc[0,1]-answer.iloc[0,1])**2+(predict.iloc[0,32]-answer.iloc[0,32])**2
#a2 = (predict.iloc[0,2]-answer.iloc[0,2])**2+(predict.iloc[0,33]-answer.iloc[0,33])**2
#a3 = (predict.iloc[0,3]-answer.iloc[0,3])**2+(predict.iloc[0,34]-answer.iloc[0,34])**2

test_nums = 4   #28p4:1 28p8:2

a31 = (predict.iloc[0,3]-answer.iloc[0,3])**2+(predict.iloc[0,34]-answer.iloc[0,62])**2
for j in range(0,1684*test_nums):   #1684*x
    
    for i in range(1,32):
        locals()['b'+str(j+1)+'_'+str(i)] = (predict.iloc[j,i]-answer.iloc[j,i])**2+(predict.iloc[j,i+31]-answer.iloc[j,i+31])**2
    
    
        locals()['a'+str(j+1)+'_'+str(i)] = locals()['b'+str(j+1)+'_'+str(i)]

#array1 = [a1_1,a1_2,a1_3,a1_4,a1_5,a1_6,a1_7,a1_8,a1_9,a1_10,a1_11,a1_12,a1_13,a1_14,a1_15,a1_16,a1_17,a1_18,a1_19,a1_20,a1_21,a1_22,a1_23,a1_24,a1_25,a1_26,a1_27,a1_28,a1_29,a1_30,a1_31]
#array2 = [a2_1,a2_2,a2_3,a2_4,a2_5,a2_6,a2_7,a2_8,a2_9,a2_10,a2_11,a2_12,a2_13,a2_14,a2_15,a2_16,a2_17,a2_18,a2_19,a2_20,a2_21,a2_22,a2_23,a2_24,a2_25,a2_26,a2_27,a2_28,a2_29,a2_30,a2_31]
#arrayn = [a[j+1]_i,a[j+1]_i+1,a[j+1]_i+2+.....+30]

for s in range(0,1684*test_nums):   #1684*x
    for k in range(1,32):
        locals()['c'+str(s+1)+'_'+str(k)] = locals()['a'+str(s+1)+'_'+str(k)]
        
    locals()['array'+str(s+1)] =  [locals()['c'+str(s+1)+'_'+str(1)],locals()['c'+str(s+1)+'_'+str(2)],
                                   locals()['c'+str(s+1)+'_'+str(3)],locals()['c'+str(s+1)+'_'+str(4)],
                                   locals()['c'+str(s+1)+'_'+str(5)],locals()['c'+str(s+1)+'_'+str(6)],
                                   locals()['c'+str(s+1)+'_'+str(7)],locals()['c'+str(s+1)+'_'+str(8)],
                                   locals()['c'+str(s+1)+'_'+str(9)],locals()['c'+str(s+1)+'_'+str(10)],
                                   locals()['c'+str(s+1)+'_'+str(11)],locals()['c'+str(s+1)+'_'+str(12)],
                                   locals()['c'+str(s+1)+'_'+str(13)],locals()['c'+str(s+1)+'_'+str(14)],
                                   locals()['c'+str(s+1)+'_'+str(15)],locals()['c'+str(s+1)+'_'+str(16)],
                                   locals()['c'+str(s+1)+'_'+str(17)],locals()['c'+str(s+1)+'_'+str(18)],
                                   locals()['c'+str(s+1)+'_'+str(19)],locals()['c'+str(s+1)+'_'+str(20)],
                                   locals()['c'+str(s+1)+'_'+str(21)],locals()['c'+str(s+1)+'_'+str(22)],
                                   locals()['c'+str(s+1)+'_'+str(23)],locals()['c'+str(s+1)+'_'+str(24)],
                                   locals()['c'+str(s+1)+'_'+str(25)],locals()['c'+str(s+1)+'_'+str(26)],
                                   locals()['c'+str(s+1)+'_'+str(27)],locals()['c'+str(s+1)+'_'+str(28)],
                                   locals()['c'+str(s+1)+'_'+str(29)],locals()['c'+str(s+1)+'_'+str(30)],
                                   locals()['c'+str(s+1)+'_'+str(31)]
            
            
            
            
            ]
    
    
finalarray = []
for t in range(0,1684*test_nums): #1684*x
    finalarray.append(locals()['array'+str(t+1)])
finalarray = DataFrame(finalarray)
a = finalarray
#a = pd.DataFrame(a)
print(a.iloc[0,1])
#print(a,len(a))

#a['row_sum'] = a.apply(lambda x:x.sum(), axis=1)
a['row_sum'] = a.iloc[:,0:31].sum(axis=1)
a.loc['total'] = a.iloc[0:1684*test_nums,31:32].sum(axis=0)   #1684*x 
print(a)
print(a.iloc[1684*test_nums,31]/(52204*test_nums))     #1684*x, 52204*x 
    
#finalarray = DataFrame(finalarray)
                                  
#finalarray.to_excel('square_error_lab224_rbf.xlsx') #linearsvr or rbf
#finalarray.to_csv('C://Users//701//Desktop//calculate_MSE//square_error_lab208_rbf.csv') #linearsvr or rbf
#finalarray.to_csv('square_error_lab209-1_linearsvr.csv') #linearsvr or rbf 
#finalarray.to_csv('C://Users//701//Desktop//calculate_MSE//square_error_lab94_linearsvr.csv')
