from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import pandas as pd

# Load and preprocess data
data_1 = pd.read_csv('F:\\Huang_ATSC\\Yuan_ML\\data\\16qam_train\\train_test.csv')
x_train = data_1.drop(['ans', 'id'], axis=1).values
y_train = data_1['ans'].values

data_3 = pd.read_csv('F:\\Huang_ATSC\\Yuan_ML\\data\\16qam_test\\test_3_15_1000\\random_feature_numbers1000_test.csv')
x_test = data_3.drop(['complex'], axis=1).values

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Build MLP model
model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000)

# Train the model
model.fit(x_train_scaled, y_train)

# Predict using the trained model
predictions = model.predict(x_test_scaled)

# Save predictions to a CSV file
predictions_df = pd.DataFrame(predictions, columns=['ans'])
predictions_df.to_csv('F:\\Huang_ATSC\\Yuan_ML\\data\\16qam_test\\test_3_15_1000\\predict_answer111_mlp.csv', index=False)
