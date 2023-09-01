from sklearn.cluster import KMeans
import pandas as pd

# Load and preprocess data
data_1 = pd.read_csv('F:\\Huang_ATSC\\MLforSchool\\data\\16qam_train\\train_3_15.csv')
x_train = data_1.drop(['ans', 'complex', 'id'], axis=1).values

data_3 = pd.read_csv('F:\\Huang_ATSC\\MLforSchool\\data\\16qam_test\\random_feature100_forTest.csv')
x_test = data_3.drop(['complex'], axis=1).values



# Build K-Means clustering model
n_clusters = 4  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=n_clusters)

# Fit the model to the data
kmeans.fit(x_train)

# Predict cluster labels for the test data
cluster_labels = kmeans.predict(x_test)

# Save cluster labels to a CSV file
cluster_labels_df = pd.DataFrame(cluster_labels, columns=['ans'])
cluster_labels_df.to_csv('F:\\Huang_ATSC\\MLforSchool\\data\\16qam_test\\ans_for_test\\cluster_labels.csv', index=False)
