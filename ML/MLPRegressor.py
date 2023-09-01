from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import pandas as pd

# train data
data_1 = pd.read_csv('F:\\Huang_ATSC\\MLforSchool\\data\\16qam_train\\train_3_15.csv')
x_train = data_1.drop(['ans','complex','id'], axis=1).values
y_train = data_1['ans'].values
# test data
data_3 = pd.read_csv('F:\\Huang_ATSC\\MLforSchool\\data\\16qam_test\\random_feature100_forTest.csv')
x_test = data_3.drop(['complex'], axis=1).values


# Build MLP model
alpha_value = 0.01
model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000, alpha=alpha_value)

# Train the model
model.fit(x_train, y_train)

# Predict the answer for test
predictions = model.predict(x_test)

# save csv
predictions_df = pd.DataFrame(predictions, columns=['ans'])
predictions_df.to_csv('F:\\Huang_ATSC\\MLforSchool\\data\\16qam_test\\ans_for_test\\mlp_predict_answer_3_15_100.csv', index=False)
