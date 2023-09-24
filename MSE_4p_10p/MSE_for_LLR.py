import pandas as pd
from sklearn.metrics import mean_squared_error
import statistics

# actual answer
actual_answers = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_randomfeature\\16qam_test\\ans_for_test\\actual_ans_10_15_100.csv') 

# predict answer
predicted_answers = pd.read_csv('D://MLforSchool//dnn_experiments//lab1_trans_answer_mlp_16qam_10_15_100_neurons15.csv')


def calc_mse(col):
    actual_values = actual_answers[col].values
    predicted_values = predicted_answers[col].values
    mse = mean_squared_error(actual_values, predicted_values)
    print("Mean Squared Error (MSE):", mse)
    return mse
# pick the answer

calc_list = map(calc_mse,['b0','b1','b2','b3'])
print(statistics.mean(calc_list))

