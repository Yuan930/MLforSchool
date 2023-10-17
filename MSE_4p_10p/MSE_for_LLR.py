import pandas as pd
from sklearn.metrics import mean_squared_error
import statistics

bit = 2

for i in range(bit):
    predicted_answers_csv_name = f"channel//mlp_answer_lab5_16qam_10_15_b{i}channel_35_70_35_20.csv"


    # actual answer
    # actual_answers = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_randomfeature\\16qam_test\\ans_for_test\\actual_ans_10_15_100.csv') 
    actual_answers = pd.read_csv(f'D:\\MLforSchool\\data\\16qam_for_channel\\16qam_test\\ans\\lab4_maximum_LLR_result_b{i}.csv') 


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
    list = [str(i) for i in range(n)]
    calc_list = map(calc_mse,list)

    print(predicted_answers_csv_name)
    print(statistics.mean(calc_list))

    # def Confirm_whether_the_plus_and_minus_signs_are_correct(a,b):
    #     apple = 0
    #     for j in range(len(a)):
            
    #         if (a[j] > 0 and b[j] > 0) or (a[j] < 0 and b[j] < 0):
    #             pass
    #         else:
    #             apple = apple+1
    #             print(j)
    #     return print(apple)
    
    # actual_answers_item = actual_answers.iloc[0:, 1:]
    # list_actual_answers_item = list(actual_answers_item.values.flatten())
    # # print(list_actual_answers_item)

    # predicted_answers_item = predicted_answers.iloc[0:, 1:]
    # list_predicted_answers_item = list(predicted_answers_item.values.flatten())
    # # print(list_predicted_answers_item)

    # check = Confirm_whether_the_plus_and_minus_signs_are_correct(list_actual_answers_item, list_predicted_answers_item)
