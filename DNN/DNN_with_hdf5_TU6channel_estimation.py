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
from tools import change_i_to_j, change_all_positive, split_real_and_imag, remove_parentheses
qam = 'qpsk'
first_nodes = 525
second_nodes =1050
third_nodes = 525
four_nodes = 300
bit = 2 # The answer of b0 or b1 ...
column = 31
dataframes = []
def feature_csv_change_to_list(feature_csv):
    feature_i_to_j = feature_csv.iloc[0:, 1:].applymap(change_i_to_j)
    feature = feature_i_to_j.applymap(change_all_positive)
    list_complex =  feature.values.flatten()
    return list(map(split_real_and_imag, list_complex))
def train_and_test_feature(Ui,H):
    all_result = []
    var = [0.1585]
    for k, row_Ui in Ui.iterrows():
            list_h = H.iloc[k].tolist()
            list_Ui = list(map(lambda x : split_real_and_imag(remove_parentheses(change_all_positive(change_i_to_j(x)))), row_Ui.tolist()[1:]))
            # print(list_Ui)
            list_h.pop(0)
            # print(list_h)
            for num in list_Ui:
                # print(num + list_h + var)
                result = num + list_h + var
                all_result.append(result)
    return all_result
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

for i in [0]:
    # for items in [9]:
    for items in [1]:
      
        test_Ui_csv = pd.read_csv(f'D:\\Desktop\\data\\check_BER\\0307yuan_QPSK_8dB_cr4_TU6\\Y_BER_8p7_29to36_fortest.csv')
        test_predictH_data_csv = pd.read_csv(f'D:\\Desktop\\data\\interpolation\\0307yuan_QPSK_8dB_cr4_TU6\\test2_8p7_29to36_fortest.csv')
        test_ans_csv = pd.read_csv(f'D:\\Desktop\\data\\check_BER\\0307yuan_QPSK_8dB_cr4_TU6\\ans_positive_LogMapllr_29to36_fortest_b{i}.csv')
        list_test_ans = ans_and_H_csv_change_to_list(test_ans_csv)
        
        test_feature = train_and_test_feature(test_Ui_csv,test_predictH_data_csv)
        
        for j in range(bit):
            
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            def transpose(list1):
                return[list(row) for row in zip(*list1)]

            # test_ans_csv = pd.read_csv(f'D:\\OneDrive - 國立臺北科技大學\\MLforSchool\\data\\256qam_for_channel\\0125_lab1_tu6\\ans\\lab1_func2_snr{snr}_256qamUi{items}_coderate10_LLR_result_b{j}.csv')

            dictionary_of_pridict_ans1 = {}
            predict_ans = []
            print(j)
            MODEL_PATH = f'D://Desktop//data//0307_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}_adam0.0001_b{i}.h5'
            model = load_model(MODEL_PATH)

            predict_ans= model.predict(test_feature) #訓練好model使用predict預測看看在訓練的model跑的回歸線
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
            csv1.T.to_csv(f'D://Desktop//data//0307_525_1050_525_300_adam0.0001_b{i}.csv')

            # predict_csv = pd.read_csv(f'D://MLforSchool//dnn_experiments//channel//1211_256qam//snr17//0111_mlp_actual_lab1_train40000_256qam{items}_10_15_LogMap_b{j}channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')
            predict_csv = pd.read_csv(f'D://Desktop//data//0307_525_1050_525_300_adam0.0001_b{i}.csv')

            n = column
            data_list = [str(i) for i in range(n)]
            calc_list = map(calc_mse,data_list)
            print(statistics.mean(calc_list))
        #     dictionary_of_pridict_ans2 = {}
        #     a = predict_csv.iloc[0:, 1:].values.flatten()

        #     if j == 0:
        #         predict_real_llr_csv = np.vectorize(modify_llr_in_bit0)(list_test_imag_part, a)

        #     elif j == 1:
        #         predict_real_llr_csv = np.vectorize(modify_llr_in_bit1)(list_test_real_part, a)
            
        #     else:
        #         predict_real_llr_csv = flatten_predict_ans
            
        #     for item in predict_real_llr_csv:
        #         if index not in dictionary_of_pridict_ans2:
        #             dictionary_of_pridict_ans2[index] = []
        #         dictionary_of_pridict_ans2[index].append(item)
        #         if len(dictionary_of_pridict_ans2[index]) >= column:
        #             index = index + 1
        #     csv2 = pd.DataFrame(dictionary_of_pridict_ans2)
        #     csv2.T.to_csv(f'D://OneDrive - 國立臺北科技大學//MLforSchool//channel//0222_TU6_in_lab1//actual_pre_llr//snr{snr}//0222_mlp_actual_lab1_num3_train40000_256qam_10_15_LogMap_b{j}channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')
            
        # dict = {}
        # a = pd.read_csv(f'D://OneDrive - 國立臺北科技大學//MLforSchool//channel//0222_TU6_in_lab1//actual_pre_llr//snr{snr}//0222_mlp_actual_lab1_num3_train40000_256qam_10_15_LogMap_b0channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')
        # b = pd.read_csv(f'D://OneDrive - 國立臺北科技大學//MLforSchool//channel//0222_TU6_in_lab1//actual_pre_llr//snr{snr}//0222_mlp_actual_lab1_num3_train40000_256qam_10_15_LogMap_b1channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')
        # c = pd.read_csv(f'D://OneDrive - 國立臺北科技大學//MLforSchool//channel//0222_TU6_in_lab1//actual_pre_llr//snr{snr}//0222_mlp_actual_lab1_num3_train40000_256qam_10_15_LogMap_b2channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')
        # d = pd.read_csv(f'D://OneDrive - 國立臺北科技大學//MLforSchool//channel//0222_TU6_in_lab1//actual_pre_llr//snr{snr}//0222_mlp_actual_lab1_num3_train40000_256qam_10_15_LogMap_b3channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')
        # e = pd.read_csv(f'D://OneDrive - 國立臺北科技大學//MLforSchool//channel//0222_TU6_in_lab1//actual_pre_llr//snr{snr}//0222_mlp_actual_lab1_num3_train40000_256qam_10_15_LogMap_b4channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')
        # f = pd.read_csv(f'D://OneDrive - 國立臺北科技大學//MLforSchool//channel//0222_TU6_in_lab1//actual_pre_llr//snr{snr}//0222_mlp_actual_lab1_num3_train40000_256qam_10_15_LogMap_b5channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')
        # g = pd.read_csv(f'D://OneDrive - 國立臺北科技大學//MLforSchool//channel//0222_TU6_in_lab1//actual_pre_llr//snr{snr}//0222_mlp_actual_lab1_num3_train40000_256qam_10_15_LogMap_b6channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')
        # h = pd.read_csv(f'D://OneDrive - 國立臺北科技大學//MLforSchool//channel//0222_TU6_in_lab1//actual_pre_llr//snr{snr}//0222_mlp_actual_lab1_num3_train40000_256qam_10_15_LogMap_b7channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')
        # list_a = list(a.iloc[0:,1:].values.flatten())
        # list_b = list(b.iloc[0:,1:].values.flatten())
        # list_c = list(c.iloc[0:,1:].values.flatten())
        # list_d = list(d.iloc[0:,1:].values.flatten())
        # list_e = list(e.iloc[0:,1:].values.flatten())
        # list_f = list(f.iloc[0:,1:].values.flatten())
        # list_g = list(g.iloc[0:,1:].values.flatten())
        # list_h = list(h.iloc[0:,1:].values.flatten())
        # # print("list_a",l。ist_a)
        # combine_list = list( item for pair in zip(list_a, list_b, list_c, list_d, list_e, list_f, list_g, list_h) for item in pair)
        # dict = combine_list

        # result = {}
        # index = 0
        # for item in dict:
        #     if index not in result:
        #         result[index] = []
        #     result[index].append(item)
        #     if (len(result[index]) >= 64800):  #根據測試資料的列數更改
        #         index = index + 1
        # # print(result)

        # llr = pd.DataFrame(result)

        # llr.T.to_csv(f'D://OneDrive - 國立臺北科技大學//MLforSchool//channel//0222_TU6_in_lab1//actual_pre_llr//snr{snr}//for_matlab//0222_mlp_actual_lab1_num3_func2_train40000_256qam{items}_10_15_LogMap_channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')
