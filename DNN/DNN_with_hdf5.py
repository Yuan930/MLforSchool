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

from tools import change_i_to_j, change_all_positive, split_real_and_imag

first_nodes = 140
second_nodes =280
third_nodes = 140
four_nodes = 80
bit = 4 # The answer of b0 or b1 ...
i = 2
column = 100
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def transpose(list1):
    return[list(row) for row in zip(*list1)]



test_feature_csv = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_channel\\16qam_test\\lab5_16qamUi_coderate10_snr8_test.csv')
test_feature_i_to_j = test_feature_csv.iloc[0:, 1:].applymap(change_i_to_j)
test_feature = test_feature_i_to_j.applymap(change_all_positive)
list_test_complex = test_feature.values.flatten()
list_test_feature = list(map(split_real_and_imag, list_test_complex))

test_ans_csv = pd.read_csv(f'D:\\MLforSchool\\data\\16qam_for_channel\\16qam_test\\ans\\lab5_Max_Log_LLR_result_b{i}.csv')

dictionary_of_pridict_ans = {}
predict_ans = []
print(i)

MODEL_PATH = 'D://MLforSchool//DNN//iris_modelb3.h5'
model = load_model(MODEL_PATH)




predict_ans= model.predict(list_test_feature) #訓練好model使用predict預測看看在訓練的model跑的回歸線
flatten_predict_ans = predict_ans.flatten()
index = 0
for item in flatten_predict_ans:
    if index not in dictionary_of_pridict_ans:
        dictionary_of_pridict_ans[index] = []
    dictionary_of_pridict_ans[index].append(item)
    if len(dictionary_of_pridict_ans[index]) >= column:
        index = index + 1

# print(dictionary_of_pridict_ans)

csv = pd.DataFrame(dictionary_of_pridict_ans)

csv.T.to_csv(f'D://MLforSchool//dnn_experiments//channel//mlp_lab6_16qam_10_15_Max_Log_b{i}channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')


predict_csv = pd.read_csv(f'D://MLforSchool//dnn_experiments//channel//mlp_lab6_16qam_10_15_Max_Log_b{i}channel_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}.csv')
def calc_mse(col):
    actual_values = test_ans_csv[col].values
    predicted_values = predict_csv[col].values
    mse = mean_squared_error(actual_values, predicted_values)
    # print("Mean Squared Error (MSE):", mse)
    return mse

n = 100
data_list = [str(i) for i in range(n)]
calc_list = map(calc_mse,data_list)
print(statistics.mean(calc_list))
