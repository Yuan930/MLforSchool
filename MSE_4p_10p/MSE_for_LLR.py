import pandas as pd
from sklearn.metrics import mean_squared_error
import statistics
import matplotlib.pyplot as plt
from tools import Extract_real_parts, Extract_imaginary_parts, change_to_complex, change_i_to_j
import numpy as np
#35_70_35_20  70_140_70_40  105_210_105_60
bit = 8
i= 3
qam = 256
snr = 17
item = 4000
# for i in range(bit):
point_h0_csv = pd.read_csv(f'D:\\MLforSchool\\data\\constellations\\{qam}qam_for_0\\{qam}qam_10_15.csv')
point_h1_csv = pd.read_csv(f'D:\\MLforSchool\\data\\constellations\\{qam}qam_for_1\\{qam}qam_10_15.csv')
receiver_point = pd.read_csv(f'D:\\MLforSchool\\data\\{qam}qam_for_channel\\{qam}qam_test\\lab1_{qam}qamUi_coderate10_snr17_4000test_positive.csv')
receiver_point_item = list(receiver_point.iloc[0:, 1:].values.flatten())

# predicted_answers_csv_name = f"channel//lab4_snr8db_20000//mlp_lab6_{qam}qam_10_15_LogMap_b{i}channel_105_210_105_60.csv"
# # predicted_answers2_csv_name = f"channel//lab4//mlp_without_valid_answer_lab4_{qam}qam_10_15_b{i}channel_70_140_70_40.csv"
# print(f'bit{i}')
# actual answer
# actual_answers = pd.read_csv('D:\\MLforSchool\\data\\{qam}qam_for_randomfeature\\{qam}qam_test\\ans_for_test\\actual_ans_10_15_100.csv') 
actual_answers = pd.read_csv(f'D:\\MLforSchool\\data\\{qam}qam_for_channel\\{qam}qam_test\\ans\\lab1_LogMap_snr{snr}_LLR_result_b{i}_{item}.csv') 
# actual_answers = pd.read_csv(f'D:\\MLforSchool\\data\\{qam}qam_for_channel\\{qam}qam_test\\ans\\lab5_MaxLog_LLR_result_b{i}.csv') 
# actual_answers = pd.read_csv(f'D://MLforSchool//dnn_experiments//{predicted_answers2_csv_name}')
# predict answer
predicted_answers = pd.read_csv(f'D://MLforSchool//dnn_experiments//channel//mlp_lab1_256qam_snr17_10_15_LogMap_b{i}channel_280_140_80.csv')
# predicted_answers = pd.read_csv(f'D:\\MLforSchool\\data\\{qam}qam_for_channel\\{qam}qam_test\\ans\\lab1_MaxLog_snr{snr}_result_b{i}_{item}.csv')


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

# print(f'{predicted_answers_csv_name}')
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

    for _ in range(50):
        max_gap = max(gaps)
        max_gap_index = gaps.index(max_gap)
        max_errors.append(max_gap)
        max_error_indices.append(max_gap_index)
        gaps[max_gap_index] = float("-inf")

    for i in range(50):
        print(f"Top {i + 1} Max Error: {max_errors[i]}")
        print(f"Receiver Point Item: {receiver_point_item[max_error_indices[i]]}")
        c.append(receiver_point_item[max_error_indices[i]])
    c_complex = [complex(item) for item in c]
    c_df = pd.DataFrame(c_complex, columns=['ComplexValue'])
    draw_receiver_point = receiver_point.iloc[0:,1:].applymap(change_to_complex)
    point_h0_complex = point_h0_csv.iloc[0:, 1:].applymap(change_i_to_j)
    point_h1_complex = point_h1_csv.iloc[0:, 1:].applymap(change_i_to_j)

    plt.figure(figsize=(5, 5))
    plt.scatter(Extract_real_parts((draw_receiver_point)), Extract_imaginary_parts(draw_receiver_point), label='test_point', marker='o', color='c', s=5)
    plt.scatter(Extract_real_parts(point_h0_complex), Extract_imaginary_parts(point_h0_complex), label='16QAM constellations', marker='o', color='r', s=5)
    plt.scatter(Extract_real_parts(c_df), Extract_imaginary_parts(c_df), label='max_error_point', marker='o', color='b', s=10)
    plt.scatter(Extract_real_parts(point_h1_complex), Extract_imaginary_parts(point_h1_complex), marker='o', color='r', s=5)

    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title("Ui' Plot")
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)  # x軸線
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)  # y軸線

    # 顯示label位置
    plt.legend(loc='upper right')
    
    # 設置xy軸範圍
    plt.xlim(0, 2.5)
    plt.ylim(0, 2.5)

    # 显示图形
    plt.show()

def max_error_plot(a, b):
    gaps = [abs(a[i] - b[i]) for i in range(len(a))]
    plt.xlim(0, 2)
    
    x = np.arange(0, 2, 0.01)
    plt.hist(gaps, bins=x, width=0.008, edgecolor='black')
    plt.xlabel('max_error')
    plt.ylabel('items')
    plt.title(f'Subtraction of actual LLR and predict LLR bit{i}')
    plt.legend()
    plt.grid(True)
    plt.show()
    
actual_answers_item = actual_answers.iloc[0:, 1:]
list_actual_answers_item = list(actual_answers_item.values.flatten())
# print(list_actual_answers_item)

predicted_answers_item = predicted_answers.iloc[0:, 1:]
list_predicted_answers_item = list(predicted_answers_item.values.flatten())
# print(list_predicted_answers_item)

# 畫圖看max error
# check = Confirm_whether_the_plus_and_minus_signs_are_correct(list_actual_answers_item, list_predicted_answers_item)
max_error = max_error_plot(list_actual_answers_item, list_predicted_answers_item)
find_top_10_max_errors(list_actual_answers_item, list_predicted_answers_item)

