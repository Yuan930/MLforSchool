import pandas as pd
from sklearn.metrics import mean_squared_error

# actual answer
actual_answers = pd.read_csv('F:\\Huang_ATSC\\MLforSchool\\data\\16qam_test\\ans_for_test\\actual_answer_3_15_100.csv') 

# predict answer
predicted_answers = pd.read_csv('F:\\Huang_ATSC\\MLforSchool\\data\\16qam_test\\ans_for_test\\mlp_predict_answer_3_15_100.csv')

# pick the answer
actual_values = actual_answers['ans'].values
predicted_values = predicted_answers['ans'].values

# calculate MSE
mse = mean_squared_error(actual_values, predicted_values)

print("Mean Squared Error (MSE):", mse)
