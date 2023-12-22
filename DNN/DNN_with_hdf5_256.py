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
qam = 256
first_nodes = 105
second_nodes =210
third_nodes = 105
four_nodes = 60
bit = 8 # The answer of b0 or b1 ...
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

for items in range(1,41):
    print("items",items)
    test_feature_csv = pd.read_csv(f'D:\\MLforSchool\\data\\{qam}qam_for_channel\\1120_lab1_r_100\\snr17\\lab1_256qamUi{items}_coderate10_snr17.csv')
    test_feature_i_to_j = test_feature_csv.iloc[0:, 1:].applymap(change_i_to_j)
    test_feature_all_positive = test_feature_i_to_j.applymap(change_all_positive)
    list_test_all_positive_complex = test_feature_all_positive.values.flatten()
    list_test_all_positive_feature = list(map(split_real_and_imag, list_test_all_positive_complex))


    list_test_complex = test_feature_i_to_j.values.flatten()
    list_complex_feature = [complex(item) for item in list_test_complex]
    list_test_imag_part = [item.imag for item in list_complex_feature]
    list_test_real_part = [item.real for item in list_complex_feature]
    for j in range(bit):
        column = 8100
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        def transpose(list1):
            return[list(row) for row in zip(*list1)]

        test_ans_csv = pd.read_csv(f'D:\\MLforSchool\\data\\{qam}qam_for_channel\\1120_lab1_r_100\\snr17\\ans\\lab1_256qamUi{items}_LogMap_snr17_LLR_result_b{j}.csv')

        dictionary_of_pridict_ans1 = {}
        predict_ans = []
        print(j)
        
        if j == 6:
            MODEL_PATH = f'D://MLforSchool//DNN//lab4_snr17_105_210_105_60_LogMap_modelb6-1_bad.h5'
        else:
            MODEL_PATH = f'D://MLforSchool//DNN//lab4_snr17_105_210_105_60_LogMap_modelb{j}.h5'

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

        csv1.T.to_csv(f'D://MLforSchool//dnn_experiments//channel//1211_256qam//snr17//mlp_lab4_{qam}qam{items}_10_15_LogMap_b{j}channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}_b6bad.csv')
        # csv1.T.to_csv('D://MLforSchool//dnn_experiments//channel//test.csv')

        predict_csv = pd.read_csv(f'D://MLforSchool//dnn_experiments//channel//1211_256qam//snr17//mlp_lab4_{qam}qam{items}_10_15_LogMap_b{j}channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}_b6bad.csv')
        # predict_csv = pd.read_csv('D://MLforSchool//dnn_experiments//channel//test.csv')

        # n = column
        # data_list = [str(i) for i in range(n)]
        # calc_list = map(calc_mse,data_list)
        # print(statistics.mean(calc_list))
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
        csv2.T.to_csv(f'D://MLforSchool//dnn_experiments//channel//1211_256qam//actual_pre_llr//snr17//mlp_actual_lab4_256qam{items}_10_15_LogMap_b{j}channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}_b6bad.csv')
        
    dict = {}
    a = pd.read_csv(f'D://MLforSchool//dnn_experiments//channel//1211_256qam//actual_pre_llr//snr17//mlp_actual_lab4_256qam{items}_10_15_LogMap_b0channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}_b6bad.csv')
    b = pd.read_csv(f'D://MLforSchool//dnn_experiments//channel//1211_256qam//actual_pre_llr//snr17//mlp_actual_lab4_256qam{items}_10_15_LogMap_b1channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}_b6bad.csv')
    c = pd.read_csv(f'D://MLforSchool//dnn_experiments//channel//1211_256qam//actual_pre_llr//snr17//mlp_actual_lab4_256qam{items}_10_15_LogMap_b2channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}_b6bad.csv')
    d = pd.read_csv(f'D://MLforSchool//dnn_experiments//channel//1211_256qam//actual_pre_llr//snr17//mlp_actual_lab4_256qam{items}_10_15_LogMap_b3channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}_b6bad.csv')
    e = pd.read_csv(f'D://MLforSchool//dnn_experiments//channel//1211_256qam//actual_pre_llr//snr17//mlp_actual_lab4_256qam{items}_10_15_LogMap_b4channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}_b6bad.csv')
    f = pd.read_csv(f'D://MLforSchool//dnn_experiments//channel//1211_256qam//actual_pre_llr//snr17//mlp_actual_lab4_256qam{items}_10_15_LogMap_b5channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}_b6bad.csv')
    g = pd.read_csv(f'D://MLforSchool//dnn_experiments//channel//1211_256qam//actual_pre_llr//snr17//mlp_actual_lab4_256qam{items}_10_15_LogMap_b6channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}_b6bad.csv')
    h = pd.read_csv(f'D://MLforSchool//dnn_experiments//channel//1211_256qam//actual_pre_llr//snr17//mlp_actual_lab4_256qam{items}_10_15_LogMap_b7channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}_b6bad.csv')
    list_a = list(a.iloc[0:,1:].values.flatten())
    list_b = list(b.iloc[0:,1:].values.flatten())
    list_c = list(c.iloc[0:,1:].values.flatten())
    list_d = list(d.iloc[0:,1:].values.flatten())
    list_e = list(e.iloc[0:,1:].values.flatten())
    list_f = list(f.iloc[0:,1:].values.flatten())
    list_g = list(g.iloc[0:,1:].values.flatten())
    list_h = list(h.iloc[0:,1:].values.flatten())
    # print("list_a",l。ist_a)
    combine_list = list( item for pair in zip(list_a, list_b, list_c, list_d, list_e, list_f, list_g, list_h) for item in pair)
    dict = combine_list

    result = {}
    index = 0
    for item in dict:
        if index not in result:
            result[index] = []
        result[index].append(item)
        if (len(result[index]) >= 64800):  #根據測試資料的列數更改
            index = index + 1
    # print(result)

    llr = pd.DataFrame(result)

    llr.T.to_csv((f'D://MLforSchool//dnn_experiments//channel//1211_256qam//actual_pre_llr//snr17//for_matlab//mlp_actual_lab4_256qam{items}_10_15_LogMap_channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}_b6bad.csv'))
