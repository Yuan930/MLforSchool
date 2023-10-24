import pandas as pd
import numpy as np

# bit = 3
# a = pd.read_csv(f'D:\\MLforSchool\\data\\16qam_for_channel\\16qam_test\\ans\\result_b{bit}.csv')

# def divide(l):
#     return l / 0.0158
bit = 4

def positive_negative(a):
    positive = sum(1 for val in a if val >0)
    negative = len(a) - positive
    return positive, negative

for i in range(bit):
    

    # print(i)
    actual_answers = pd.read_csv(f'D:\\MLforSchool\\data\\16qam_for_channel\\16qam_valid\\ans\\lab4_maximum_LLR_result_b{i}.csv') 
    list_actual_answers_item = list(actual_answers.iloc[0:, 1:].values.flatten())

    predict_answers = pd.read_csv(f'D://MLforSchool//dnn_experiments//channel//mlp_answer_lab4_16qam_10_15_b{i}channel_35_70_35_20.csv') 
    list_predict_answers_item = list(predict_answers.iloc[0:, 1:].values.flatten())

    actual_positive, actual_negative = positive_negative(list_actual_answers_item)
    predict_positive, predict_negative= positive_negative(list_predict_answers_item)

    print(f'bit{i}\n正數:{actual_positive}\n負數:{actual_negative}')
    print(f'bit{i}\n正數:{predict_positive}\n負數:{predict_negative}')
    
    
    
