import pandas as pd
import numpy as np

# bit = 3
# a = pd.read_csv(f'D:\\MLforSchool\\data\\16qam_for_channel\\16qam_test\\ans\\result_b{bit}.csv')
def positive_negative(a):
    negative = len(a)
    positive = 0
    for j in range(len(a)):
                
        if (a[j] > 0):
            pass
        else:
            positive = positive+1
            # print(j)
    print(f'正數',positive,'\n',f'負數',negative-positive,'\n')



# def divide(l):
#     return l / 0.0158
bit = 4
for i in range(bit):
    def positive_negative(a):
        b = len(a)
        positive = 0
        for j in range(len(a)):
                    
            if (a[j] > 0):
                pass
            else:
                positive = positive+1
            # print(j)
        negative = b - positive
        print(f'bit{i}\n正數:{positive}\n負數:{negative}')

    # print(i)
    actual_answers = pd.read_csv(f'D:\\MLforSchool\\data\\16qam_for_channel\\16qam_valid\\ans\\lab4_maximum_LLR_result_b{i}.csv') 
    actual_answers_item = actual_answers.iloc[0:, 1:]
    list_actual_answers_item = list(actual_answers_item.values.flatten())

    predict_answers = pd.read_csv(f'D://MLforSchool//dnn_experiments//channel//mlp_answer_lab5_16qam_10_15_b{i}channel_35_70_35_20.csv') 
    predict_answers_item = predict_answers.iloc[0:, 1:]
    list_predict_answers_item = list(predict_answers_item.values.flatten())

    actual_cal = positive_negative(list_actual_answers_item)
    predict_cal = positive_negative(list_predict_answers_item)


