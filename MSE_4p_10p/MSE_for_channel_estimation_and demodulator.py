# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 20:14:57 2021

@author: oscar
"""
from sklearn.metrics import mean_squared_error
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
from tools import Extract_real_parts, Extract_imaginary_parts, change_to_complex, change_i_to_j

qam = 16

def mse(df1, df2):
    # 計算兩個DataFrame之間的差的平方
    squared_diff = (df1 - df2) ** 2
    # 取平均值得到MSE
    mse_value = np.mean(squared_diff)
    return mse_value

for i in range(1):
    point_h0_csv = pd.read_csv(f'D:\\MLforSchool\\data\\constellations\\{qam}qam_for_0\\10_15\\{qam}qam_bit{i}.csv')
    point_h1_csv = pd.read_csv(f'D:\\MLforSchool\\data\\constellations\\{qam}qam_for_1\\10_15\\{qam}qam_bit{i}.csv')
    receiver_point = pd.read_csv(f'D:\\py3710\\py_train_data\\0509yuan_16qam_17dB_cr10_TU6\\test_TU6_16qam_perfect_yy_12dB_30_positive.csv')
    receiver_point_item = list(receiver_point.iloc[0:, 1:].values.flatten())
    # predict_csv = pd.read_csv(f'D:\\py3710\\py_train_data\\0509yuan_16qam_17dB_cr10_TU6\\dnn_experiments\\0510_210_420_210_120_adadelta_epoch2000_batch100_bit0_preLLR.csv')
    # predict_csv = predict_csv*5
    # predict_csv = pd.read_csv(f'D:\\py3710\\py_train_data\\0425yuan_16qam_8dB_cr10_TU6\\dnn_experiments\\0425_525_1050_525_300_adam0.0001_batch100_bit{i}_preLLR.csv')
    # predict_csv = pd.read_csv(f'D:\\py3710\\py_train_data\\0425yuan_16qam_8dB_cr10_TU6\\dnn_experiments\\0425_525_1050_525_300_adam0.0001_epoch1500_batch150_bit{i}_preLLR.csv')
    predict_csv = pd.read_csv(f'D:\\py3710\\py_train_data\\0509yuan_16qam_17dB_cr10_TU6\\test_data_MaxLogllr_b{i}.csv')
    predicted_answers_llr = list(predict_csv.iloc[0:, 1:].values.flatten())
    print(predicted_answers_llr)
    predict_ans = predict_csv.drop(['id'],axis = 1)
    # print(predict_ans)


    def transpose(list1):
        return[list(row) for row in zip(*list1)]



    answer_csv = pd.read_csv(f'D:\\py3710\\py_train_data\\0509yuan_16qam_17dB_cr10_TU6\\test_data_LogMapllr_b{i}.csv')
    ancwer_ans = answer_csv.drop(['id'],axis = 1)
    actual_answers_llr = list(answer_csv.iloc[0:, 1:].values.flatten())
    actual_answers_llr = np.clip(actual_answers_llr,-20,20).tolist()
    # print(ancwer_ans)

    mse_value = mse(predict_ans,ancwer_ans)
    # print(mse_value)

    MSE_ans = np.mean(mse_value)
    print(f"MSE_ans: {MSE_ans:.6f}")
    
    def calc_mse(actual_values, predicted_values):
    # Check for NaN and replace with a default value (e.g., 0)
        actual_values = np.nan_to_num(actual_values, nan=0)
        predicted_values = np.nan_to_num(predicted_values, nan=0)
        
        # Check for infinity and replace with a large finite value (e.g., 1e9)
        actual_values = np.where(np.isinf(actual_values), 1e9, actual_values)
        predicted_values = np.where(np.isinf(predicted_values), 1e9, predicted_values)
        
        mse = mean_squared_error(actual_values, predicted_values)
        return mse
    actual_answers_llr_array = np.array(actual_answers_llr)
    predicted_answers_llr_array = np.array(predicted_answers_llr)
    cal_MSE_list = np.mean((actual_answers_llr_array-predicted_answers_llr_array)**2)
    # print(cal_MSE_list)
    
    
    def Confirm_whether_the_plus_and_minus_signs_are_correct(a,b):
        apple = 0
        for j in range(len(a)):
            
            if (a[j] > 0 and b[j] > 0) or (a[j] < 0 and b[j] < 0):
                pass
            else:
                apple = apple+1
                print(receiver_point_item[j])
                print(f'actual: {a[j]} predict: {b[j]}')
        return print(apple)

    def find_max_error(a, b):
        gaps = [abs(a[i] - b[i]) for i in range(len(a))]
        sorted_gaps = sorted(gaps, reverse=True)
        
        max_gap = max(gaps)
        max_gap_index = gaps.index(max_gap)
        max_gap_item = receiver_point_item[max_gap_index]
        print(f"max_gap_index: {max_gap_index}, max_gap_item: {max_gap_item}")
        # 将最大差值设为负无穷，以便找到第二大差值
        gaps[max_gap_index] = float("-inf")
        
        # 找到第二大差值
        second_max_gap = max(gaps)
        second_max_gap_index = gaps.index(second_max_gap)
        print(receiver_point_item[second_max_gap_index])
        # 将第二大差值设为负无穷，以便找到第三大差值
        gaps[second_max_gap_index] = float("-inf")
        
        # 找到第三大差值
        third_max_gap = max(gaps)
        third_max_gap_index = gaps.index(third_max_gap)
        print(receiver_point_item[third_max_gap_index])
        print(max_gap_index,second_max_gap_index,third_max_gap_index)
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(sorted_gaps)), sorted_gaps)
        plt.xlabel("item")
        plt.ylabel("error")
        plt.title("The error of each item")
        
        plt.show()
        

    def find_top_10_max_errors(a, b):
        c = []
        df = []
        gaps = [abs(a[i] - b[i]) for i in range(len(a))]
        max_errors = []
        max_error_indices = []

        for _ in range(15):
            max_gap = max(gaps)
            max_gap_index = gaps.index(max_gap)
            max_errors.append(max_gap)
            max_error_indices.append(max_gap_index)
            gaps[max_gap_index] = float("-inf")

        for j in range(15):
            print(f"Top {j + 1} Max Error: {max_errors[j]:.6f}")
            print(f"Receiver Point Item: {receiver_point_item[max_error_indices[j]]}")
            print(f"LLR_ans : {actual_answers_llr[max_error_indices[j]]:.6f}")
            print(f"DNN_LLR : {predicted_answers_llr[max_error_indices[j]]:.6f}")
            c.append(receiver_point_item[max_error_indices[j]])
        print(c)    
        c_complex = [complex(item) for item in c]
        c_df = pd.DataFrame(c_complex, columns=['ComplexValue'])
        draw_receiver_point = receiver_point.iloc[0:,1:].applymap(change_to_complex)
        point_h0_complex = point_h0_csv.iloc[0:, 1:].applymap(change_i_to_j)
        point_h1_complex = point_h1_csv.iloc[0:, 1:].applymap(change_i_to_j)
        # point_train_feature_csv = train_feature_csv.iloc[0:, 1:].applymap(change_i_to_j)

        plt.figure(figsize=(5, 5))
        # plt.scatter(Extract_real_parts(point_train_feature_csv), Extract_imaginary_parts(point_train_feature_csv), label='train_point', marker='o', color='lightgray', s=5)
        plt.scatter(Extract_real_parts((draw_receiver_point)), Extract_imaginary_parts(draw_receiver_point), label='test_point', marker='o', color='c', s=5)
        plt.scatter(Extract_real_parts(point_h0_complex), Extract_imaginary_parts(point_h0_complex), label=f'{qam}QAM bit{i}=0', marker='o', color='r', s=5)
        plt.scatter(Extract_real_parts(point_h1_complex), Extract_imaginary_parts(point_h1_complex), label=f'{qam}QAM bit{i}=1', marker='o', color='m', s=5)
        plt.scatter(Extract_real_parts(c_df), Extract_imaginary_parts(c_df), label=f'bit{i}max_error_point', marker='o', color='b', s=10)
        ###########標出20與-20的點#############
        for idx, llr in enumerate(a):
            if llr >= 20 or llr <= -20:
                value = receiver_point_item[idx]
                df.append(value)
        df_complex = [complex(item) for item in df]
        find_df = pd.DataFrame(df_complex, columns=['ComplexValue'])        
        plt.scatter(Extract_real_parts(find_df), Extract_imaginary_parts(find_df), color='black', s=5)


        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.title(f"Top 15 Max Error Ui' Points in bit{i}")
        plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)  # x軸線
        plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)  # y軸線

        # 顯示label位置
        plt.legend(loc='upper right')
        
        # 設置xy軸範圍
        plt.xlim(-8, 8)
        plt.ylim(-8, 8)

        # 显示图形
        # save_path = f'G:\\我的雲端硬碟\\MLforSchool\meeting\\pic\\0425\\max_error_point_0425_525_1050_525_300_adam0.0001_epoch1500_batch150_bit{i}.png'
        # plt.savefig(save_path)
        plt.show()

    def max_error_plot(a, b):
        gaps = [abs(a[i] - b[i]) for i in range(len(a))]
        plt.xlim(0, 5)
        
        x = np.arange(0, 5, 0.01)
        plt.hist(gaps, bins=x, width=0.008, edgecolor='black')
        plt.xlabel('max_error')
        plt.ylabel('items')
        plt.title(f'Maximum Error between actual LLR and predict LLR in bit{i}')
        plt.legend()
        plt.grid(True)
        # save_path = f'G:\\我的雲端硬碟\\MLforSchool\meeting\\pic\\0425\\max_error_0425_525_1050_525_300_adam0.0001_epoch1500_batch150_bit{i}.png'
        # plt.savefig(save_path)
        plt.show()
        
        
        
    actual_answers_item = answer_csv.iloc[0:, 1:]
    list_actual_answers_item = list(actual_answers_item.values.flatten())
    # print(list_actual_answers_item)

    predicted_answers_item = predict_csv.iloc[0:, 1:]
    list_predicted_answers_item = list(predicted_answers_item.values.flatten())
    # print(list_predicted_answers_item)

    # 畫圖看max error
    # check = Confirm_whether_the_plus_and_minus_signs_are_correct(list_actual_answers_item, list_predicted_answers_item)
    max_error = max_error_plot(list_actual_answers_item, list_predicted_answers_item)
    find_top_10_max_errors(list_actual_answers_item, list_predicted_answers_item)