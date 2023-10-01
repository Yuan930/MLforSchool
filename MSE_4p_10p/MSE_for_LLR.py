import pandas as pd
from sklearn.metrics import mean_squared_error
import statistics


predicted_answers_csv_name = "channel//mlp_predict_answer_lab1_16qam_10_15_100_channel_senior.csv"


# actual answer
# actual_answers = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_channel\\16qam_test\\ans_for_test\\actual_ans_10_15_100_1.4.csv') 
actual_answers = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_channel\\16qam_test\\ans\\result_b2.csv') 


# predict answer
predicted_answers = pd.read_csv(f'D://MLforSchool//dnn_experiments//{predicted_answers_csv_name}')


def calc_mse(col):
    actual_values = actual_answers[col].values
    predicted_values = predicted_answers[col].values
    mse = mean_squared_error(actual_values, predicted_values)
    print("Mean Squared Error (MSE):", mse)
    return mse
# pick the answer
# for random system
# calc_list = map(calc_mse,['b0','b1','b2','b3'])

# for channel system
n = 10
list = [str(i) for i in range(n)]
calc_list = map(calc_mse,list)




print(predicted_answers_csv_name)
print(statistics.mean(calc_list))


