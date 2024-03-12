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

first_nodes = 525
second_nodes =1050
third_nodes = 525
four_nodes = 300

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

train_channel = '1to28'
test_channel = '29to36'
all_bit_mse_average = []
all_sorted_results_dict = {}
all_results_dict = {}

for i in [1]:
    
    mean_results = []
    mean_results_cal = []
    for j in [1]:
        ########################################TRAIN#######################################
        train_Ui_csv = pd.read_csv(f'D:\\Desktop\\data\\check_BER\\0307yuan_QPSK_8dB_cr4_TU6\\Y_BER_8p7_{train_channel}_fortrain.csv')
        Ui_train = train_Ui_csv[str(i)].values
        Ui_train_real_imag = feature_csv_change_to_list(Ui_train)
        print(Ui_train_real_imag[0])
        train_predictH_data_csv = pd.read_csv(f'D:\\Desktop\\data\\interpolation\\0307yuan_QPSK_8dB_cr4_TU6\\test2_8p7_{train_channel}_fortrain.csv')
        pilot_train = train_predictH_data_csv.drop(['id'],axis = 1).values
        print(pilot_train)
        train_ans_csv = pd.read_csv(f'D:\\Desktop\\data\\check_BER\\0307yuan_QPSK_8dB_cr4_TU6\\ans_positive_LogMapllr_{train_channel}_fortrain_b{i}.csv')
        train_ans = train_ans_csv[str(i)].values
        # print(len(train_ans))
        list_train_ans = ans_and_H_csv_change_to_list(train_ans_csv)
        list_train_ans = np.clip(list_train_ans, -20, 20).tolist()

        train_feature = train_and_test_feature(train_Ui_csv,train_predictH_data_csv)
        # print(len(train_feature))
#         #########################################TEST#####################################
#         test_Ui_csv = pd.read_csv(f'D:\\Desktop\\data\\check_BER\\0307yuan_QPSK_8dB_cr4_TU6\\Y_BER_8p7_{test_channel}_fortest.csv')
#         test_predictH_data_csv = pd.read_csv(f'D:\\Desktop\\data\\interpolation\\0307yuan_QPSK_8dB_cr4_TU6\\test2_8p7_{test_channel}_fortest.csv')
#         test_ans_csv = pd.read_csv(f'D:\\Desktop\\data\\check_BER\\0307yuan_QPSK_8dB_cr4_TU6\\ans_positive_MaxLopllr_29to36_fortest_b{i}.csv')
#         list_test_ans = ans_and_H_csv_change_to_list(test_ans_csv)
        
#         test_feature = train_and_test_feature(test_Ui_csv,test_predictH_data_csv)
#         dictionary_of_pridict_ans = {}
#         predict_ans = []
#         print("bit%d,第%d次"% (i,j))



#         model = Sequential()#進行建造網路架構 在Sequential()裡面定義層數、激勵函數
#         model.add(Dense(first_nodes, input_dim=33,  kernel_initializer='normal',activation='relu'))                               #加入神經層第一層(輸入14)輸出128 初始化器傳入 激活函數用relu #這邊的input一定要隨著特徵數量更改(pilot數乘2 因為實 虛 分開)
#         model.add(Dense(second_nodes, input_dim=first_nodes,  kernel_initializer='normal',activation='relu'))
#         model.add(Dense(third_nodes, input_dim=second_nodes,  kernel_initializer='normal',activation='relu'))
#         model.add(Dense(four_nodes, input_dim=third_nodes,  kernel_initializer='normal',activation='relu'))
#         model.add(Dense(1,  kernel_initializer='normal',activation='linear'))
#         optimizer = Adam(learning_rate=0.00001)
#         model.compile(loss='MSE', optimizer=optimizer, metrics = ['mse'])#設定model的loss和優化器(分別是MSE和adam) ,metrics=['mse','mape']
#         epochs = 100#代表疊帶40次(總共要用全部的訓練樣本重複跑幾回合)
#         batch_size = 100#為你的输入指定一个固定的 batch 大小(每個iteration以100筆做計算)

#         # history = model.fit(list_x_train_feature, list_y_train_ans, batch_size=batch_size, epochs=epochs ,verbose=1,validation_data=(list_x_valid_feature, list_y_valid_ans))
#         history = model.fit(train_feature, list_train_ans, batch_size=batch_size, epochs=epochs ,verbose=1)
#         # model.add(Dense(32, input_dim=x_train.shape[1],  kernel_initializer='normal',activation='relu'))
#         print("Saving model to disk \n")
#         mp = f"D://Desktop//data//0307_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}_adam0.0001_b{i}_realLLR.h5"
#         model.save(mp)    #存model
        
#         ##########################畫圖###################################
#         # summarize history for loss plt.plot(history.history['loss']) plt.plot(history.history['val_loss']) plt.title('model loss')
#         loss = history.history['loss']
#         # # val_loss = history.history['val_loss']

#         epochs = range(1, len(loss) + 1)
#         start_epoch = 0
#         plt.figure()
#         # plt.plot(epochs[start_epoch:],val_loss[start_epoch:], label = 'val_loss')
#         # plt.plot(epochs[start_epoch:], loss[start_epoch:], label='loss')
#         # # smoothed_loss = smooth_curve(loss)
#         # # plt.plot(epochs[start_epoch:], smoothed_loss[start_epoch:], label='Smoothed Train Loss')
#         # plt.title('model loss')
#         # plt.ylabel('MSE')
#         # plt.xlabel('epoch')


#         plt.yscale('log')  # Set y-axis to logarithmic scale
#         plt.plot(epochs[start_epoch:], loss[start_epoch:], label='loss')
#         plt.title('model loss')
#         plt.ylabel('MSE (log scale)')
#         plt.xlabel('epoch')
        
#         plt.legend(['train_loss'], loc='upper right') 
#         save_path = f'D://Desktop//data//0307_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}_adam0.0001_bit{i}_realLLR.png'
#         plt.savefig(save_path)
#         # plt.show()
#         #######################################################################################################
#         predict_ans = np.clip(model.predict(test_feature), -20, 20) #訓練好model使用predict預測看看在訓練的model跑的回歸線
#         flatten_predict_ans = predict_ans.flatten()
#         index = 0
#         for item in flatten_predict_ans:
#             if index not in dictionary_of_pridict_ans:
#                 dictionary_of_pridict_ans[index] = []
#             dictionary_of_pridict_ans[index].append(item)
#             if len(dictionary_of_pridict_ans[index]) >= column:
#                 index = index + 1

#         # print(dictionary_of_pridict_ans)

#         csv = pd.DataFrame(dictionary_of_pridict_ans)

#         csv.T.to_csv(f'D://Desktop//data//0307_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}_adam0.0001_b{i}_realLLR.csv')

#         predict_csv = pd.read_csv(f'D://Desktop//data//0307_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}_adam0.0001_b{i}_realLLR.csv')
        
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



