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
from tools import change_i_to_j, change_all_positive, split_real_and_imag

first_nodes = 105
second_nodes =210
third_nodes = 105
four_nodes = 60
snr = 17
bit = 8 # The answer of b0 or b1 ...
i = 0
column = 200
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def transpose(list1):
    return[list(row) for row in zip(*list1)]
def feature_csv_change_to_list(feature_csv):
    feature_i_to_j = feature_csv.iloc[0:, 1:].applymap(change_i_to_j)
    feature = feature_i_to_j.applymap(change_all_positive)
    list_complex =  feature.values.flatten()
    return list(map(split_real_and_imag, list_complex))
# 在損失函數中忽略 inf 值
def custom_loss(y_true, y_pred):
    mask = tf.math.is_finite(y_true)  # 判斷是否有限
    y_true = tf.where(mask, y_true, 0)  # 將非有限的值設置為 0
    y_pred = tf.where(mask, y_pred, 0)  # 將非有限的值設置為 0
    return tf.keras.losses.mean_squared_error(y_true, y_pred)


all_bit_mse_average = []
all_sorted_results_dict = {}
all_results_dict = {}
for i in range(8):
    
    mean_results = []
    mean_results_cal = []

    
    for j in range(1):
        train_feature_csv = pd.read_csv('D:\\MLforSchool\\data\\256qam_for_channel\\TU6_256qam_train\\lab1_TU6_cr10_snr17_to_21_Ui1_to_4_20000train.csv')
        list_x_train_feature = feature_csv_change_to_list(train_feature_csv)
        # print(list_x_train_feature)
        train_H_csv = pd.read_csv(f'D:\\MLforSchool\\data\\256qam_for_channel\\TU6_256qam_train\\lab1_TU6_cr10_snr17_to_21_Ui1_to_4_squaredH_divided_by_2var_20000train.csv')
        y_train_H = train_H_csv.iloc[0: ,1:]
        list_y_train_H = list(y_train_H.values.flatten())
        
        train_ans_csv = pd.read_csv(f'D:\\MLforSchool\\data\\256qam_for_channel\\TU6_256qam_train\\ans\\lab1_snr17_to_21_Ui1_to_4_LLR_result_b{i}.csv')
        z_train_ans = train_ans_csv.iloc[0: ,1:]
        list_z_train_ans = list(z_train_ans.values.flatten())

        test_ans_csv = pd.read_csv(f'D:\\MLforSchool\\data\\256qam_for_channel\\TU6_256qam_test\\ans\\lab1_snr17_to_21_Ui5_to_8_LLR_result_b{i}.csv')
        test_feature_csv = pd.read_csv('D:\\MLforSchool\\data\\256qam_for_channel\\TU6_256qam_test\\lab1_TU6_cr10_snr17_to_21_Ui5_to_8_20000test.csv')
        list_test_feature = feature_csv_change_to_list(test_feature_csv)
        test_H_csv = pd.read_csv(f'D:\\MLforSchool\\data\\256qam_for_channel\\TU6_256qam_test\\lab1_TU6_cr10_snr17_to_21_Ui5_to_8_squaredH_divided_by_2var_20000test.csv')
        test_H = test_H_csv.iloc[0: ,1:]
        list_test_H = list(test_H.values.flatten())

        dictionary_of_pridict_ans = {}
        predict_ans = []
        print("bit%d,第%d次"% (i,j+1))
        combined_input = np.column_stack((list_x_train_feature, list_y_train_H)).tolist()
        combined_output = np.column_stack((list_test_feature, list_test_H)).tolist()
        # print(combined_input)

        model = Sequential()#進行建造網路架構 在Sequential()裡面定義層數、激勵函數
        model.add(Dense(first_nodes, input_dim=3,  kernel_initializer='normal',activation='relu'))                               #加入神經層第一層(輸入14)輸出128 初始化器傳入 激活函數用relu #這邊的input一定要隨著特徵數量更改(pilot數乘2 因為實 虛 分開)
        model.add(Dense(second_nodes, input_dim=first_nodes,  kernel_initializer='normal',activation='relu'))
        model.add(Dense(third_nodes, input_dim=second_nodes,  kernel_initializer='normal',activation='relu'))
        model.add(Dense(four_nodes, input_dim=third_nodes,  kernel_initializer='normal',activation='relu'))
        model.add(Dense(1,  kernel_initializer='normal',activation='linear'))
        optimizer = Adam(learning_rate=0.0001)
        model.compile(loss=custom_loss, optimizer=optimizer, metrics = ['mse'])#設定model的loss和優化器(分別是MSE和adam) ,metrics=['mse','mape']
        epochs = 5000#代表疊帶40次(總共要用全部的訓練樣本重複跑幾回合)
        batch_size = 100#為你的输入指定一个固定的 batch 大小(每個iteration以100筆做計算)

        # history = model.fit(list_x_train_feature, list_y_train_ans, batch_size=batch_size, epochs=epochs ,verbose=1,validation_data=(list_x_valid_feature, list_y_valid_ans))
        history = model.fit(combined_input, list_z_train_ans, batch_size=batch_size, epochs=epochs ,verbose=1)
        # model.add(Dense(32, input_dim=x_train.shape[1],  kernel_initializer='normal',activation='relu'))
        print("Saving model to disk \n")
        mp = f"D://MLforSchool//DNN//0111_snr{snr}_{first_nodes}_{second_nodes}_{third_nodes}_{four_nodes}_learning_rate0.0001_b{i}_time1.h5"
        model.save(mp)    #存model
        
        ##########################畫圖###################################
        # summarize history for loss plt.plot(history.history['loss']) plt.plot(history.history['val_loss']) plt.title('model loss')
        loss = history.history['loss']
        # # val_loss = history.history['val_loss']

        epochs = range(1, len(loss) + 1)
        start_epoch = 100
        plt.figure()
        # plt.plot(epochs[start_epoch:],val_loss[start_epoch:], label = 'val_loss')
        plt.plot(epochs[start_epoch:], loss[start_epoch:], label='loss')
        # smoothed_loss = smooth_curve(loss)
        # plt.plot(epochs[start_epoch:], smoothed_loss[start_epoch:], label='Smoothed Train Loss')
        plt.title('model loss')
        plt.ylabel('MSE')
        plt.xlabel('epoch')
        
        plt.legend(['train_loss'], loc='upper right') 
        save_path = f'D://MLforSchool//DNN//dnn_loss_pic//TU6lab1train20000bit{i}_leaerninig_rate0.001.png'
        plt.savefig(save_path)
        # plt.show()
        #######################################################################################################
        predict_ans= model.predict(combined_output) #訓練好model使用predict預測看看在訓練的model跑的回歸線
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

        # csv.T.to_csv(f'D://MLforSchool//dnn_experiments//channel//1211_256qam//snr17//mlp_lab1_256qam_snr17_10_15_LogMap_b{i}channel_{first_nodes}_{second_nodes}_{third_nodes}.csv')

        # predict_csv = pd.read_csv(f'D://MLforSchool//dnn_experiments//channel//1211_256qam//snr17//mlp_lab1_256qam_snr17_10_15_LogMap_b{i}channel_{first_nodes}_{second_nodes}_{third_nodes}.csv')
        
        csv.T.to_csv(f'D://MLforSchool//dnn_experiments//channel//test.csv')

        predict_csv = pd.read_csv(f'D://MLforSchool//dnn_experiments//channel//test.csv')
        def calc_mse(col):
            actual_values = test_ans_csv[col].values
            predicted_values = predict_csv[col].values
            mse = mean_squared_error(actual_values, predicted_values)
            # print("Mean Squared Error (MSE):", mse)
            return mse

        
        
        data_list = [str(i) for i in range(column)]
        calc_list = map(calc_mse,data_list)
        mean_result = statistics.mean(calc_list)
        print(j)
        print(mean_result)
        mean_results_cal.append(mean_result)
        
    all_results_dict[f"bit{i}"] = mean_results_cal
    a = sorted(mean_results_cal)
    print("Sorted Mean Results:", a)
    all_sorted_results_dict[f"bit{i}"] = a  # 將排序結果存入字典
    average_mean_result = statistics.mean(mean_results_cal)

    print("平均MSE:", average_mean_result)
    all_bit_mse_average.append(average_mean_result)
    
# 在所有迴圈結束後顯示整體的 Sorted Mean Results 和平均 MSE
for i, mse_result in enumerate(all_bit_mse_average):
    print(f"bit{i}, 平均MSE: {mse_result}")

print("5次mse結果:")    
for key, value in all_sorted_results_dict.items():
    print(f"{key}: {value}")

# 印出每個 i 對應的排序結果
print("各個 i 對應的排序結果:")
for key, value in all_results_dict.items():
    print(f"{key}: {value}")


