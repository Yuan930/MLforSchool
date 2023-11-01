import pandas as pd
from sklearn.metrics import mean_squared_error
import statistics
import matplotlib.pyplot as plt
from tools import Extract_real_parts, Extract_imaginary_parts, change_to_complex, change_i_to_j
import numpy as np
#35_70_35_20  70_140_70_40
bit = 4
i=0
point16_h0_csv = pd.read_csv('D:\\MLforSchool\\data\\constellations\\16qam_for_0\\16qam_10_15.csv')
point16_h1_csv = pd.read_csv('D:\\MLforSchool\\data\\constellations\\16qam_for_1\\16qam_10_15.csv')
receiver_point = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_channel\\16qam_test\\lab4_16qamUi_coderate10_snr8_test_positive.csv')
receiver_point_item = list(receiver_point.iloc[0:, 1:].values.flatten())
for i in range(bit):
    predicted_answers_csv_name = f"channel//mlp_lab4_16qam_10_15_Max_Log_b{i}channel_140_280_140_80.csv"
    predicted_answers2_csv_name = f"channel//lab4//mlp_without_valid_answer_lab4_16qam_10_15_b{i}channel_70_140_70_40.csv"
    print(f'bit{i}')
    # actual answer
    # actual_answers = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_randomfeature\\16qam_test\\ans_for_test\\actual_ans_10_15_100.csv') 
    actual_answers = pd.read_csv(f'D:\\MLforSchool\\data\\16qam_for_channel\\16qam_test\\ans\\lab4_maximum_LLR_result_b{i}.csv') 
    # actual_answers = pd.read_csv(f'D:\\MLforSchool\\data\\16qam_for_channel\\16qam_test\\ans\\lab4_Max_Log_LLR_result_b{i}.csv') 
    # actual_answers = pd.read_csv(f'D://MLforSchool//dnn_experiments//{predicted_answers2_csv_name}')
    # predict answer
    predicted_answers = pd.read_csv(f'D://MLforSchool//dnn_experiments//{predicted_answers2_csv_name}')
    # predicted_answers = pd.read_csv(f'D:\\MLforSchool\\data\\16qam_for_channel\\16qam_test\\ans\\lab4_maximum_LLR_result_b{i}.csv')


    def calc_mse(col):
        actual_values = actual_answers[col].values
        predicted_values = predicted_answers[col].values
        mse = mean_squared_error(actual_values, predicted_values)
        # print("Mean Squared Error (MSE):", mse)
        return mse
    # pick the answer
    # for random system
    # calc_list = map(calc_mse,['b0','b1','b2','b3'])

    # for channel system
    n = 100
    data_list = [str(i) for i in range(n)]
    calc_list = map(calc_mse,data_list)

    print(f'{predicted_answers2_csv_name}')
    print(statistics.mean(calc_list))

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
        gaps = [abs(a[i] - b[i]) for i in range(len(a))]
        max_errors = []
        max_error_indices = []

        for _ in range(15):
            max_gap = max(gaps)
            max_gap_index = gaps.index(max_gap)
            max_errors.append(max_gap)
            max_error_indices.append(max_gap_index)
            gaps[max_gap_index] = float("-inf")

        for i in range(15):
            print(f"Top {i + 1} Max Error: {max_errors[i]}")
            print(f"Receiver Point Item: {receiver_point_item[max_error_indices[i]]}")
            c.append(receiver_point_item[max_error_indices[i]])
        c_complex = [complex(item) for item in c]
        c_df = pd.DataFrame(c_complex, columns=['ComplexValue'])
        draw_receiver_point = receiver_point.iloc[0:,1:].applymap(change_to_complex)
        point16_h0_complex = point16_h0_csv.iloc[0:, 1:].applymap(change_i_to_j)
        point16_h1_complex = point16_h1_csv.iloc[0:, 1:].applymap(change_i_to_j)

        plt.figure(figsize=(5, 5))
        plt.scatter(Extract_real_parts((draw_receiver_point)), Extract_imaginary_parts(draw_receiver_point), label='test_point', marker='o', color='c', s=5)
        plt.scatter(Extract_real_parts(point16_h0_complex), Extract_imaginary_parts(point16_h0_complex), label='16QAM constellations', marker='o', color='r', s=5)
        plt.scatter(Extract_real_parts(c_df), Extract_imaginary_parts(c_df), label='max_error_point', marker='o', color='b', s=10)
        plt.scatter(Extract_real_parts(point16_h1_complex), Extract_imaginary_parts(point16_h1_complex), marker='o', color='r', s=5)

        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.title("Ui' Plot")
        plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)  # x軸線
        plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)  # y軸線

        # 顯示label位置
        plt.legend(loc='upper right')
        
        # 設置xy軸範圍
        plt.xlim(-0.2, 2.5)
        plt.ylim(-0.2, 2.5)

        # 显示图形
        plt.show()

       
    actual_answers_item = actual_answers.iloc[0:, 1:]
    list_actual_answers_item = list(actual_answers_item.values.flatten())
    # print(list_actual_answers_item)

    predicted_answers_item = predicted_answers.iloc[0:, 1:]
    list_predicted_answers_item = list(predicted_answers_item.values.flatten())
    # print(list_predicted_answers_item)

    # check = Confirm_whether_the_plus_and_minus_signs_are_correct(list_actual_answers_item, list_predicted_answers_item)
    max_error = find_max_error(list_actual_answers_item, list_predicted_answers_item)
    find_top_10_max_errors(list_actual_answers_item, list_predicted_answers_item)

