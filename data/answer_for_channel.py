import pandas as pd
import numpy as np
import re
# 16point for h0 or h1
point16_h0 = pd.read_csv('D:\\Desktop\\MLforSchool\\data\\constellations\\16qam_for_0\\16qam_10_15.csv')
point16_h1 = pd.read_csv('D:\\Desktop\\MLforSchool\\data\\constellations\\16qam_for_1\\16qam_10_15.csv')
#random_feature = pd.read_csv('D:\\MLforSchool\\data\\16qam_test\\random_feature100_forTest.csv')
#random_feature = pd.read_csv('D:\\MLforSchool\\data\\16qam_train\\random_feature1000_forTrain.csv')
random_feature = pd.read_csv('D:\\Desktop\\MLforSchool\\data\\QPSK_for_channel\\U.csv')


point16_h0.replace('i', 'j', regex=True, inplace=True)
point16_h1.replace('i', 'j', regex=True, inplace=True)
random_feature.replace('i', 'j', regex=True, inplace=True)

def point16_h(row):
    return [complex(row[str(j)]) for j in range(1, 9)]

results = [[] for _ in range(4)]

for j, _ in random_feature.iterrows():
    d0 = []
    d1 = []
    for i, row_h0 in point16_h0.iterrows():
        h0 = point16_h(row_h0)    
        d0_row = (abs(data - complex(val)) for data, val in zip(h0, range(1, 11)))  # for b = 0
        d0.append(min(d0_row))
    for i, row_h1 in point16_h1.iterrows():
        h1 = point16_h(row_h1)
        d1_row = (abs(data - complex(val)) for data, val in zip(h1, range(1,11)))  # for b = 1
        d1.append(min(d1_row))
    for i in range(len(d0)):
        results.append(d1[i] - d0[i])
        
        
        fUi = pd.DataFrame(results[i], columns=['ans'])
        fUi.to_csv(f'D:\\Desktop\\MLforSchool\\data\\QPSK_for_channel\\result_i_{i}.csv', index=False)

        


        

