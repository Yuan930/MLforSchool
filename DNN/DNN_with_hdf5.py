# -*- coding: utf-8 -*-
"""
Created on Sun Oct 1 20:12:12 2023

@author: Ting Yuan Huang
"""
from sklearn.metrics import mean_squared_error
from keras.models import load_model
from keras.layers import *
from keras.callbacks import *
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import DataFrame
import os
import statistics
from tools import change_i_to_j, change_all_positive, split_real_and_imag

first_nodes = 140
second_nodes =280
third_nodes = 140
four_nodes = 80
bit = 4 # The answer of b0 or b1 ...
dataframes = []

def calc_mse(col):
        actual_values = test_ans_csv[col].values
        predicted_values = predict_csv[col].values
        mse = mean_squared_error(actual_values, predicted_values)
        # print("Mean Squared Error (MSE):", mse)
        return mse
    
def modify_llr_in_bit0(imaginary_part, llr):
    if imaginary_part < 0:
       return -llr
    return llr

def modify_llr_in_bit1(real_part, llr):
    if real_part < 0:
       return -llr
    return llr
test_feature_csv = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_channel\\16qam_test\\lab5_16qam_1015_snr8_20000test.csv')
test_feature_i_to_j = test_feature_csv.iloc[0:, 1:].applymap(change_i_to_j)
test_feature_all_positive = test_feature_i_to_j.applymap(change_all_positive)
list_test_all_positive_complex = test_feature_all_positive.values.flatten()
list_test_all_positive_feature = list(map(split_real_and_imag, list_test_all_positive_complex))


list_test_complex = test_feature_i_to_j.values.flatten()
list_complex_feature = [complex(item) for item in list_test_complex]
list_test_imag_part = [item.imag for item in list_complex_feature]
list_test_real_part = [item.real for item in list_complex_feature]
for j in range(bit):
    column = 1000
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    def transpose(list1):
        return[list(row) for row in zip(*list1)]

    test_ans_csv = pd.read_csv(f'D:\\MLforSchool\\data\\16qam_for_channel\\16qam_test\\ans\\lab5_maximum_LLR_result_b{j}.csv')

    dictionary_of_pridict_ans1 = {}
    predict_ans = []
    print(j)

    MODEL_PATH = f'D://MLforSchool//DNN//snr8_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}_modelb{j}.h5'
    model = load_model(MODEL_PATH)




    predict_ans= model.predict(list_test_all_positive_feature) #訓練好model使用predict預測看看在訓練的model跑的回歸線
    flatten_predict_ans = predict_ans.flatten()
    index = 0
    for item in flatten_predict_ans:
        if index not in dictionary_of_pridict_ans1:
            dictionary_of_pridict_ans1[index] = []
        dictionary_of_pridict_ans1[index].append(item)
        if len(dictionary_of_pridict_ans1[index]) >= column:
            index = index + 1

    # print(dictionary_of_pridict_ans1)

    csv1 = pd.DataFrame(dictionary_of_pridict_ans1)

    csv1.T.to_csv(f'D://MLforSchool//dnn_experiments//channel//lab6//mlp_lab6_16qam_10_15_LogMap_b{j}channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')


    predict_csv = pd.read_csv(f'D://MLforSchool//dnn_experiments//channel//lab6//mlp_lab6_16qam_10_15_LogMap_b{j}channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')
    

    n = column
    data_list = [str(i) for i in range(n)]
    calc_list = map(calc_mse,data_list)
    print(statistics.mean(calc_list))
    dictionary_of_pridict_ans2 = {}
    a = predict_csv.iloc[0:, 1:].values.flatten()

    if j == 0:
        predict_real_llr_csv = np.vectorize(modify_llr_in_bit0)(list_test_imag_part, a)

    elif j == 1:
        predict_real_llr_csv = np.vectorize(modify_llr_in_bit1)(list_test_real_part, a)
    
    else:
        predict_real_llr_csv = flatten_predict_ans
    
    for item in predict_real_llr_csv:
        if index not in dictionary_of_pridict_ans2:
            dictionary_of_pridict_ans2[index] = []
        dictionary_of_pridict_ans2[index].append(item)
        if len(dictionary_of_pridict_ans2[index]) >= column:
            index = index + 1
    csv2 = pd.DataFrame(dictionary_of_pridict_ans2)
    csv2.T.to_csv(f'D://MLforSchool//dnn_experiments//channel//lab6//actual_pre_llr//mlp_actual_lab6_16qam_10_15_LogMap_b{j}channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')
    

a = pd.read_csv(f'D://MLforSchool//dnn_experiments//channel//lab6//actual_pre_llr//mlp_actual_lab6_16qam_10_15_LogMap_b0channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')
b = pd.read_csv(f'D://MLforSchool//dnn_experiments//channel//lab6//actual_pre_llr//mlp_actual_lab6_16qam_10_15_LogMap_b1channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')
c = pd.read_csv(f'D://MLforSchool//dnn_experiments//channel//lab6//actual_pre_llr//mlp_actual_lab6_16qam_10_15_LogMap_b2channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')
d = pd.read_csv(f'D://MLforSchool//dnn_experiments//channel//lab6//actual_pre_llr//mlp_actual_lab6_16qam_10_15_LogMap_b3channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')
combine_frames = [pd.concat([a.iloc[0:, 1:], b.iloc[0:, 1:], c.iloc[0:, 1:], d.iloc[0:, 1:]], axis=1)]
llr = pd.concat(combine_frames, axis=1)
llr.to_csv((f'D://MLforSchool//dnn_experiments//channel//lab6//actual_pre_llr//mlp_actual_lab6_16qam_10_15_LogMap_channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv'))
print(combine_frames)
print(llr)