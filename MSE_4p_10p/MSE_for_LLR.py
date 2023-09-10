import pandas as pd
from sklearn.metrics import mean_squared_error

# actual answer
actual_answers = pd.read_csv('D:\\MLforSchool\\data\\16qam_test\\ans_for_test\\actual_ans_10_15_100.csv') 

# predict answer
predicted_answers = pd.read_csv('D://MLforSchool//dnn_experiments//lab1_trans_answer_mlp_16qam_10_15_100.csv')

# pick the answer
for col in ['0', '1', '2', '3']:
    actual_values = actual_answers[col].values
    predicted_values = predicted_answers[col].values

# calculate MSE
mse = mean_squared_error(actual_values, predicted_values)

print("Mean Squared Error (MSE):", mse)