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
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import DataFrame
import os
import re
import tensorflow as tf
from keras import backend as K
from tools import change_i_to_j, change_all_positive, remove_parentheses, split_real_and_imag
from keras.models import load_model
first_nodes = 210
second_nodes =420
third_nodes = 210
four_nodes = 120

bit = 2
i = 0
column = 31
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def transpose(list1):
    return[list(row) for row in zip(*list1)]
def feature_csv_change_to_list(feature_csv):
    feature_i_to_j = np.vectorize(change_i_to_j)(feature_csv)
    feature = np.vectorize(change_all_positive)(feature_i_to_j)
    list_complex = feature.flatten()
    return list(map(split_real_and_imag, list_complex))

def ans_and_H_csv_change_to_list(ans_and_H_csv):
    ans_H = ans_and_H_csv.iloc[0: ,1:]
    list_ans_H = list(ans_H.values.flatten())
    return list_ans_H
def train_and_test_feature(Ui,pilot):
    var = [0.1585]
    result = [x + y + [z] for x, y, z in zip(Ui, pilot, var)]
    return result

def complex_to_positive(complex_str):
    real_imag_parts = complex_str.replace('i', 'j').replace('+', ' ').split()

    if 'j' not in real_imag_parts[0]:
        real_part = float(real_imag_parts[0]) if real_imag_parts[0] else 0.0
        imag_part = 0.0
    else:
        real_part, imag_part = map(float, real_imag_parts[0].split('j'))

    positive_real = abs(real_part)
    positive_imag = abs(imag_part)

    return positive_real, positive_imag


all_bit_mse_average = []
all_sorted_results_dict = {}
all_results_dict = {}
batch_size = 100
for i in [0]:
    
    mean_results = []
    mean_results_cal = []
    for j in range(1,32):

#         #########################################TEST#####################################
        test_Ui_csv = pd.read_csv(f'D:\\py3710\\py_train_data\\0509yuan_16qam_17dB_cr10_TU6\\test_TU6_16qam_Y_BER_8p7_12dB_30.csv')
        Ui_test = test_Ui_csv[str(j)].values
        Ui_test_real_imag = np.array(feature_csv_change_to_list(Ui_test))
        
        test_predictH_data_csv = pd.read_csv(f'D:\\py3710\\py_train_data\\0509yuan_16qam_17dB_cr10_TU6\\test_pilot_8p7_TU6_16qam_12dB_30.csv')
        pilot_test = test_predictH_data_csv.drop(['id'],axis = 1).values
        var_test = np.full((Ui_test_real_imag.shape[0], 1), 0.0631)
        # test_ans_csv = pd.read_csv(f'D:\\Desktop\\data\\check_BER\\0307yuan_QPSK_8dB_cr4_TU6\\ans_positive_LogMapllr_29to36_fortest_b{i}.csv')
        # test_ans = test_ans_csv.drop(['id'],axis =1).values

        test_feature = np.hstack((Ui_test_real_imag, pilot_test,var_test))
#         dictionary_of_pridict_ans = {}
#         predict_ans = []
#         print("bit%d,第%d次"% (i,j))
        epochs = 2000#代表疊帶40次(總共要用全部的訓練樣本重複跑幾回合)
        batch_size = 100

        MODEL_PATH = f"D:\\py3710\\py_train_data\\0509yuan_16qam_17dB_cr10_TU6\\DNN_hdf5\\0510_(-20,20)_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}_adadelta_epoch{epochs}_batch{batch_size}_bit{i}_preLLR{j}.h5"
        model = load_model(MODEL_PATH)
        

#         #######################################################################################################
        locals()['anss'+str(j)] = model.predict(test_feature) #訓練好model使用predict預測看看在訓練的model跑的回歸線
        locals()['anss'+str(j)] = np.clip(locals()['anss'+str(j)], -20, 20)
        print(i,'column',j)
    arr = [anss1,anss2,anss3,anss4,anss5,anss6,anss7,anss8,anss9,anss10,anss11,anss12,anss13,anss14,anss15,anss16,anss17,anss18,anss19,anss20,anss21,anss22,anss23,anss24,anss25,anss26,anss27,anss28,anss29,anss30,anss31]
    print(arr)

    print(arr)
    print(arr[0][1])
    print(arr[1][1])
    print(arr[0][1]-arr[1][1])
    a = transpose(arr)
    a = DataFrame(a)
    a.to_csv(f'D:\\py3710\\py_train_data\\0509yuan_16qam_17dB_cr10_TU6\\DNN_predict_ans\\0510_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}_adadelta_epoch2000_batch100_bit{i}_preLLR.csv')
#         # print(dictionary_of_pridict_ans)

#         csv = pd.DataFrame(dictionary_of_pridict_ans)

#         csv.T.to_csv(f'D://Desktop//data//0307_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}_adadelta_b{i}_realLLR.csv')

#         predict_csv = pd.read_csv(f'D://Desktop//data//0307_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}_adadelta_b{i}_realLLR.csv')
        
#         # csv.T.to_csv(f'D://MLforSchool//dnn_experiments//channel//test.csv')

#         # predict_csv = pd.read_csv(f'D://MLforSchool//dnn_experiments//channel//test.csv')
#         def calc_mse(col):
#             actual_values = test_ans_csv[col].values
#             predicted_values = predict_csv[col].values
#             mse = mean_squared_error(actual_values, predicted_values)
#             # print("Mean Squared Error (MSE):", mse)
#             return mse

        
        
#     #     data_list = [str(i) for i in range(column)]
#     #     calc_list = map(calc_mse,data_list)
#     #     mean_result = statistics.mean(calc_list)
#     #     print(j)
#     #     print(mean_result)
#     #     mean_results_cal.append(mean_result)
#     # print('MSE',mean_results_cal)    
#     # all_results_dict[f"bit{i}"] = mean_results_cal
#     # a = sorted(mean_results_cal)
#     # print("Sorted Mean Results:", a)
# # #     all_sorted_results_dict[f"bit{i}"] = a  # 將排序結果存入字典
# # #     average_mean_result = statistics.mean(mean_results_cal)

# # #     print("平均MSE:", average_mean_result)
# # #     all_bit_mse_average.append(average_mean_result)
    
# # # # 在所有迴圈結束後顯示整體的 Sorted Mean Results 和平均 MSE
# # # for i, mse_result in enumerate(all_bit_mse_average):
# # #     print(f"bit{i}, 平均MSE: {mse_result}")

# # # print("5次mse結果:")    
# # # for key, value in all_sorted_results_dict.items():
# # #     print(f"{key}: {value}")

# # # # 印出每個 i 對應的排序結果
# # # print("各個 i 對應的排序結果:")
# # # for key, value in all_results_dict.items():
# # #     print(f"{key}: {value}")



