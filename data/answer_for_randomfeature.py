import pandas as pd
import numpy as np
import re
# 16point for h0 or h1
point16_h0 = pd.read_csv('D:\\MLforSchool\\data\\constellations\\16qam_for_0\\16qam_10_15.csv')
point16_h1 = pd.read_csv('D:\\MLforSchool\\data\\constellations\\16qam_for_1\\16qam_10_15.csv')
#random_feature = pd.read_csv('D:\\MLforSchool\\data\\16qam_test\\random_feature100_forTest.csv')
#random_feature = pd.read_csv('D:\\MLforSchool\\data\\16qam_train\\random_feature1000_forTrain.csv')
random_feature = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_randomfeature\\16qam_valid\\random_feature100_forVaild.csv')


point16_h0.replace('i', 'j', regex=True, inplace=True)
point16_h1.replace('i', 'j', regex=True, inplace=True)

def point16_h(row):
    return [complex(row[str(j)]) for j in range(1, 9)]

results = []
for j, row_rf in random_feature.iterrows():
    d0 = []
    d1 = []
    for i, row_h0 in point16_h0.iterrows():
        h0 = point16_h(row_h0)    
        d0_row = (abs(data - complex(row_rf['complex'])) for data in h0)      #for bk = 0
        d0.append(min(d0_row))
        print(row_rf)
    for i, row_h1 in point16_h1.iterrows():
        h1 = point16_h(row_h1)
        d1_row = (abs(data - complex(row_rf['complex'])) for data in h1)      #for bk = 1
        d1.append(min(d1_row))
    
    for i in range(len(d0)):
        results.append(d1[i] - d0[i])

result_matrix = []
# 每行四個答案
num_rows = len(results) // 4
result_matrix = np.array(results[:4 * num_rows]).reshape(num_rows, 4)
fUi = pd.DataFrame(result_matrix, columns=[f'b{i}' for i in range(4)])

#generate_fUi = fUi.to_csv('D:\\MLforSchool\\data\\16qam_train\\train_10_15_1000.csv') #train
#generate_fUi = fUi.to_csv('D:\\MLforSchool\\data\\16qam_test\\ans_for_test\\actual_ans_10_15_100.csv') #test
#generate_fUi = fUi.to_csv('D:\\MLforSchool\\data\\16qam_for_randomfeature\\16qam_valid\\ans\\valid_10_15_100000.csv') #valid



# data1 = pd.read_csv('E:\\Huang_ATSC\\MLforSchool\\data\\constellations\\16qam_16point.csv')
# #random_feature = pd.read_csv('E:\\Huang_ATSC\\MLforSchool\\data\\random_feature100_forTest.csv')
# random_feature = pd.read_csv('E:\\Huang_ATSC\\MLforSchool\\data\\16qam_train\\random_feature1000_forTrain.csv')

# def point16_complex_for0(row):
#     return [complex(row[str(j)]) for j in range(1, 17)]

# complex_data = {}
# for i, row in data1.iterrows():
#     if i >= 1:
#         index = int(row['id'].split('_')[0])
#         complex_data[index] = point16_complex_for0(row)

# results = []
# for j, row in random_feature.iterrows():
#     d0 = []
#     d1 = []
#     for data in complex_data[2]:
#         i = complex_data[2].index(data)
        
#         if i % 2 == 0:
#             d0.append(abs(data - complex(row['complex'])))      #for bk = 0
#         else:
#             d1.append(abs(data - complex(row['complex'])))      #for bk = 1
                
        
#     aa = min(d1) - min(d0)
#     #print(min(d1))
#     results.append(aa)
#     fUi = pd.DataFrame(results, columns=['ans'])
    
#     generate_fUi = fUi.to_csv('F:\\Huang_ATSC\\MLforSchool\\data\\16qam_train\\train_2_15.csv') #train
#     #generate_fUi = fUi.to_csv('F:\\Huang_ATSC\\MLforSchool\\data\\16qam_test\\ans_for_test\\test_3_15_100.csv') #test

