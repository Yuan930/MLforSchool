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
first_nodes = 210
second_nodes =420
third_nodes = 210
four_nodes = 120
bit = 8 # The answer of b0 or b1 ...
dataframes = []
def feature_csv_change_to_list(feature_csv):
    feature_i_to_j = feature_csv.iloc[0:, 1:].applymap(change_i_to_j)
    feature = feature_i_to_j.applymap(change_all_positive)
    list_complex =  feature.values.flatten()
    return list(map(split_real_and_imag, list_complex))

def ans_and_H_csv_change_to_list(ans_and_H_csv):
    ans_H = ans_and_H_csv.iloc[0: ,1:]
    list_ans_H = list(ans_H.values.flatten())
    return list_ans_H

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

for snr in [17,18,19,20,21]:
    # for items in [9]:
    for items in [1,2,3,4,5,6,7,8,10]:
        print("snr",snr,"items",items)
        test_feature_csv = pd.read_csv(f'D:\\MLforSchool\\data\\256qam_for_channel\\0125_lab1_tu6\\lab1_snr{snr}_256qamUi{items}_coderate10.csv')
        test_feature_i_to_j = test_feature_csv.iloc[0:, 1:].applymap(change_i_to_j)
        test_feature_all_positive = test_feature_i_to_j.applymap(change_all_positive)
        list_test_all_positive_complex = test_feature_all_positive.values.flatten()
        list_test_all_positive_feature = list(map(split_real_and_imag, list_test_all_positive_complex))
        
        test_H_csv = pd.read_csv(f'D:\\MLforSchool\\data\\256qam_for_channel\\0125_lab1_tu6\\squaredH_divided_by_2var\\lab1_snr{snr}_256qamUi{items}_coderate10_squaredH_divided_by_2var_real.csv')
        list_test_H = ans_and_H_csv_change_to_list(test_H_csv)

        combined_input = np.column_stack((list_test_all_positive_feature, list_test_H)).tolist()

        list_test_complex = test_feature_i_to_j.values.flatten()
        list_complex_feature = [complex(item) for item in list_test_complex]
        list_test_imag_part = [item.imag for item in list_complex_feature]
        list_test_real_part = [item.real for item in list_complex_feature]
        
        for j in range(bit):
            column = 8100
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            def transpose(list1):
                return[list(row) for row in zip(*list1)]

            # test_ans_csv = pd.read_csv(f'D:\\OneDrive - 國立臺北科技大學\\MLforSchool\\data\\256qam_for_channel\\0125_lab1_tu6\\ans\\lab1_func2_snr{snr}_256qamUi{items}_coderate10_LLR_result_b{j}.csv')

            dictionary_of_pridict_ans1 = {}
            predict_ans = []
            print(j)
            # MODEL_PATH = f'D://MLforSchool//DNN//0119_TU6_CR10_DnnModel//0222_TU6lab2_func2_train40000_snr17_210_420_210_120_adam0.0001_b{j}_time7.h5'
            if j == 0 or j == 1 or j == 3:
                MODEL_PATH = f'D://MLforSchool//DNN//0119_TU6_CR10_DnnModel//0222_TU6lab2_func2_train40000_snr17_105_210_105_60_adadelta_b{j}_time3.h5'
            else:    
                MODEL_PATH = f'D://MLforSchool//DNN//0119_TU6_CR10_DnnModel//0222_TU6lab2_func2_train40000_snr17_105_210_105_60_adadelta_b{j}_time4.h5'
            model = load_model(MODEL_PATH)

            predict_ans= model.predict(combined_input) #訓練好model使用predict預測看看在訓練的model跑的回歸線
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

            # csv1.T.to_csv(f'D://MLforSchool//dnn_experiments//channel//1211_256qam//snr17//0111_mlp_actual_lab1_train40000_256qam{items}_10_15_LogMap_b{j}channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')
            csv1.T.to_csv('D://MLforSchool//dnn_experiments//channel//test.csv')

            # predict_csv = pd.read_csv(f'D://MLforSchool//dnn_experiments//channel//1211_256qam//snr17//0111_mlp_actual_lab1_train40000_256qam{items}_10_15_LogMap_b{j}channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')
            predict_csv = pd.read_csv('D://MLforSchool//dnn_experiments//channel//test.csv')

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
            csv2.T.to_csv(f'D://OneDrive - 國立臺北科技大學//MLforSchool//channel//0222_TU6_in_lab1//actual_pre_llr//snr{snr}//0222_mlp_actual_lab1_num3_train40000_256qam_10_15_LogMap_b{j}channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')
            
        dict = {}
        a = pd.read_csv(f'D://OneDrive - 國立臺北科技大學//MLforSchool//channel//0222_TU6_in_lab1//actual_pre_llr//snr{snr}//0222_mlp_actual_lab1_num3_train40000_256qam_10_15_LogMap_b0channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')
        b = pd.read_csv(f'D://OneDrive - 國立臺北科技大學//MLforSchool//channel//0222_TU6_in_lab1//actual_pre_llr//snr{snr}//0222_mlp_actual_lab1_num3_train40000_256qam_10_15_LogMap_b1channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')
        c = pd.read_csv(f'D://OneDrive - 國立臺北科技大學//MLforSchool//channel//0222_TU6_in_lab1//actual_pre_llr//snr{snr}//0222_mlp_actual_lab1_num3_train40000_256qam_10_15_LogMap_b2channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')
        d = pd.read_csv(f'D://OneDrive - 國立臺北科技大學//MLforSchool//channel//0222_TU6_in_lab1//actual_pre_llr//snr{snr}//0222_mlp_actual_lab1_num3_train40000_256qam_10_15_LogMap_b3channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')
        e = pd.read_csv(f'D://OneDrive - 國立臺北科技大學//MLforSchool//channel//0222_TU6_in_lab1//actual_pre_llr//snr{snr}//0222_mlp_actual_lab1_num3_train40000_256qam_10_15_LogMap_b4channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')
        f = pd.read_csv(f'D://OneDrive - 國立臺北科技大學//MLforSchool//channel//0222_TU6_in_lab1//actual_pre_llr//snr{snr}//0222_mlp_actual_lab1_num3_train40000_256qam_10_15_LogMap_b5channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')
        g = pd.read_csv(f'D://OneDrive - 國立臺北科技大學//MLforSchool//channel//0222_TU6_in_lab1//actual_pre_llr//snr{snr}//0222_mlp_actual_lab1_num3_train40000_256qam_10_15_LogMap_b6channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')
        h = pd.read_csv(f'D://OneDrive - 國立臺北科技大學//MLforSchool//channel//0222_TU6_in_lab1//actual_pre_llr//snr{snr}//0222_mlp_actual_lab1_num3_train40000_256qam_10_15_LogMap_b7channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')
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

        llr.T.to_csv(f'D://OneDrive - 國立臺北科技大學//MLforSchool//channel//0222_TU6_in_lab1//actual_pre_llr//snr{snr}//for_matlab//0222_mlp_actual_lab1_num3_func2_train40000_256qam{items}_10_15_LogMap_channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')
