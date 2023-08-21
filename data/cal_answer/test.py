

import pandas as pd
import re
# 16point
data1 = pd.read_csv('E:\\Huang_ATSC\\Yuan_ML\\train_data\\16qam_16point.csv')
data2 = pd.read_csv('E:\\Huang_ATSC\\Yuan_ML\\train_data\\random_complex_numbers100.csv')

# 固定的複數
# fixed_complex = 1 + 1j

complex_data = {}
# Function to extract complex numbers from a row
def point16_complex(row):
    return [complex(row[str(j)]) for j in range(1, 17, 2)]

#16 point
for i, row1 in data1.iterrows():
        if i >= 1:
            index = int(row1['id'].split('_')[0])
            # 將複數儲存在字典中，使用 i 作為鍵
            complex_data[index] = point16_complex(row1)

# 計算每個複數和固定複數之間的距離並儲存到字典中

results = []
# 從第2行開始
for j, row2 in data2.iterrows():
    min_distance = float('inf')
    for complex_num in complex_data[3]:      # w = 2_16~13_16
        distance = abs(complex_num - complex(row2['complex']))
        print('distance',distance)
        if distance < min_distance:
            min_distance = distance
    print('min_distance',min_distance)
    results.append(min_distance)
    
answer = pd.DataFrame(results, columns=['ans'])
aa = answer.to_csv('E:\\Huang_ATSC\\Yuan_ML\\train_data\\d0.csv')    
print(complex_data[3])

