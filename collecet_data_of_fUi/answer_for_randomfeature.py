import pandas as pd
import numpy as np
import re
# 16point for h0 or h1
point16_h0_csv = pd.read_csv('D:\\MLforSchool\\data\\constellations\\16qam_for_0\\16qam_10_15.csv')
point16_h1_csv = pd.read_csv('D:\\MLforSchool\\data\\constellations\\16qam_for_1\\16qam_10_15.csv')
random_feature_csv = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_randomfeature\\16qam_test\\random_feature100_forTest.csv')
#random_feature = pd.read_csv('D:\\MLforSchool\\data\\16qam_train\\random_feature1000_forTrain.csv')
#random_feature_csv = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_randomfeature\\16qam_valid\\random_feature100_forVaild.csv')


point16_h0_csv.replace('i', 'j', regex=True, inplace=True)
point16_h1_csv.replace('i', 'j', regex=True, inplace=True)

def point16_h(row):
    return [complex(row[str(j)]) for j in range(1, 9)]
def cal_distance(a,b):
    return abs(a-b)

results = []
for j, row_rf in random_feature_csv.iterrows():
    d0 = []
    d1 = []
    
    def cal_distance_of_random_feature_item(h):
        return min(cal_distance(data, complex(row_rf['complex'])) for data in h)
    
    for i, row_h0 in point16_h0_csv.iterrows():
        h0 = point16_h(row_h0)         
        d0.append(cal_distance_of_random_feature_item(h0))  #for bk = 0
        
    for i, row_h1 in point16_h1_csv.iterrows():
        h1 = point16_h(row_h1)       
        d1.append(cal_distance_of_random_feature_item(h1))  #for bk = 1
    
    for i in range(len(d0)):
        results.append(d1[i] - d0[i])
     
#整理成想要的資料存放形式 
bit = 4     #根據bit數量做更改
results_matrix = {}
index = 0
for item in results:   
    if index not in results_matrix:
        results_matrix[index] = []
    results_matrix[index].append(item)
    if (len(results_matrix[index]) >= bit):
        index = index + 1
print(results_matrix)        
    
       

# result_matrix = []
# # 每行四個答案
# num_rows = len(results) // 4
# result_matrix = np.array(results[:4 * num_rows]).reshape(num_rows, 4)
csv = pd.DataFrame(results_matrix)
csv.index = [f'b{i}' for i in range(bit)]
#generate_fUi = fUi.to_csv('D:\\MLforSchool\\data\\16qam_train\\train_10_15_1000.csv') #train
csv.T.to_csv('D:\\MLforSchool\\data\\16qam_for_randomfeature\\16qam_test\\ans_for_test\\actual_ans_10_15_1001.csv') #test
#csv.T.to_csv('D:\\MLforSchool\\data\\16qam_for_randomfeature\\16qam_valid\\valid_10_15_1000.csv') #valid


