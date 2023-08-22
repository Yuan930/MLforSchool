import pandas as pd
import re
# 16point
data1 = pd.read_csv('F:\\Huang_ATSC\\MLforSchool\\data\\constellations\\16qam_16point.csv')
#random_feature = pd.read_csv('F:\\Huang_ATSC\\MLforSchool\\data\\random_feature100_forTest.csv')
random_feature = pd.read_csv('F:\\Huang_ATSC\\MLforSchool\\data\\16qam_train\\random_feature1000_forTrain.csv')

def point16_complex_for0(row):
    return [complex(row[str(j)]) for j in range(1, 17)]

complex_data = {}
for i, row in data1.iterrows():
    if i >= 1:
        index = int(row['id'].split('_')[0])
        complex_data[index] = point16_complex_for0(row)

results = []
for j, row in random_feature.iterrows():
    d0 = []
    d1 = []
    for data in complex_data[2]:
        i = complex_data[2].index(data)
        
        if i % 2 == 0:
            d0.append(abs(data - complex(row['complex'])))      #for bk = 0
        else:
            d1.append(abs(data - complex(row['complex'])))      #for bk = 1
                
        
    aa = min(d1) - min(d0)
    #print(min(d1))
    results.append(aa)
    fUi = pd.DataFrame(results, columns=['ans'])
    
    generate_fUi = fUi.to_csv('F:\\Huang_ATSC\\MLforSchool\\data\\16qam_train\\train_2_15.csv') #train
    #generate_fUi = fUi.to_csv('F:\\Huang_ATSC\\MLforSchool\\data\\16qam_test\\ans_for_test\\test_3_15_100.csv') #test
