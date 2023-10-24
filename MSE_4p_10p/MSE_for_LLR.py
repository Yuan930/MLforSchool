import pandas as pd
from sklearn.metrics import mean_squared_error
import statistics
import matplotlib.pyplot as plt

bit = 4
receiver_point = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_channel\\16qam_test\\lab4_16qamUi_coderate10_snr8_test.csv')
receiver_point_item = list(receiver_point.iloc[0:, 1:].values.flatten())
for i in range(bit):
    predicted_answers_csv_name = f"channel//mlp_lab4_16qam_10_15_Max_Log_b{i}channel_35_70_35_20.csv"
    # predicted_answers_csv_name = f"channel//lab4//mlp_without_valid_answer_lab4_16qam_10_15_b{i}channel_35_70_35_20.csv"
    print(f'bit{i}')
    # actual answer
    # actual_answers = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_randomfeature\\16qam_test\\ans_for_test\\actual_ans_10_15_100.csv') 
    # actual_answers = pd.read_csv(f'D:\\MLforSchool\\data\\16qam_for_channel\\16qam_test\\ans\\lab4_maximum_LLR_result_b{i}.csv') 
    actual_answers = pd.read_csv(f'D:\\MLforSchool\\data\\16qam_for_channel\\16qam_test\\ans\\lab4_Max_Log_LLR_result_b{i}.csv') 

    # predict answer
    predicted_answers = pd.read_csv(f'D://MLforSchool//dnn_experiments//{predicted_answers_csv_name}')
    # predicted_answers = pd.read_csv(f'D:\\MLforSchool\\data\\16qam_for_channel\\16qam_test\\ans\\MAX_LOG_LLR_result_b{bit}.csv')


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

    print(predicted_answers_csv_name)
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
        
        # 将最大差值设为负无穷，以便找到第二大差值
        gaps[max_gap_index] = float("-inf")
        
        # 找到第二大差值
        second_max_gap = max(gaps)
        second_max_gap_index = gaps.index(second_max_gap)
        
        # 将第二大差值设为负无穷，以便找到第三大差值
        gaps[second_max_gap_index] = float("-inf")
        
        # 找到第三大差值
        third_max_gap = max(gaps)
        third_max_gap_index = gaps.index(third_max_gap)
        print(max_gap_index,second_max_gap_index,third_max_gap_index)
        print(max_gap, second_max_gap, third_max_gap)
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(sorted_gaps)), sorted_gaps)
        plt.xlabel("item")
        plt.ylabel("error")
        plt.title("The error of each item")
        plt.show()
    
    actual_answers_item = actual_answers.iloc[0:, 1:]
    list_actual_answers_item = list(actual_answers_item.values.flatten())
    # print(list_actual_answers_item)

    predicted_answers_item = predicted_answers.iloc[0:, 1:]
    list_predicted_answers_item = list(predicted_answers_item.values.flatten())
    # print(list_predicted_answers_item)

    # check = Confirm_whether_the_plus_and_minus_signs_are_correct(list_actual_answers_item, list_predicted_answers_item)
    # max_error = find_max_error(list_actual_answers_item, list_predicted_answers_item)


##檢查座標點時，列+1，行-1