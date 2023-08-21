import pandas as pd
import re
# 16point
data1 = pd.read_csv('F:\\Huang_ATSC\\Yuan_ML\\data\\16qam_16point.csv')
random_feature = pd.read_csv('F:\\Huang_ATSC\\Yuan_ML\\data\\16qam_train\\random_feature_numbers100_test.csv')

def point16_complex_for0(row):
    return [complex(row[str(j)]) for j in range(1, 17)]

complex_data = {}
for i, row in data1.iterrows():
    if i >= 1:
        index = int(row['id'].split('_')[0])
        # 將複數儲存在字典中，使用 i 作為鍵
        complex_data[index] = point16_complex_for0(row)

results = []
for j, row in random_feature.iterrows():
    d0 = []
    d1 = []
    for data in complex_data[3]:
        i = complex_data[3].index(data)
        
        if i % 2 == 0:
            d0.append(abs(data - complex(row['complex'])))
        else:
            d1.append(abs(data - complex(row['complex'])))
                
        
    aa = min(d1) - min(d0)
    #print(min(d1))
    results.append(aa)
    fUi = pd.DataFrame(results, columns=['ans'])
    generate_fUi = fUi.to_csv('F:\\Huang_ATSC\\Yuan_ML\\data\\16qam_train\\train_test.csv')

