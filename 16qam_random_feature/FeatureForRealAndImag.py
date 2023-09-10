import pandas as pd
import numpy as np
import random
import re

# 生成隨機複數
def generate_complex_number():
    x_real = random.uniform(0, 3)
    y_imag = random.uniform(0, 3)
    return complex(x_real, y_imag)

# 設定負數數量
num_complex_numbers = 100

complex_numbers = []

# 生成複數
for i in range(num_complex_numbers):
    complex_number = generate_complex_number()
    complex_numbers.append(complex_number)


data = pd.DataFrame(complex_numbers, columns=['complex'])

# 將複數的實部和虛部分開，分別儲存在不同的欄位中
data['feature1'] = data['complex'].apply(lambda x: x.real)
data['feature2'] = data['complex'].apply(lambda x: x.imag)
data['complex'] = data['complex'].apply(lambda x: re.sub(r'[()]', '', str(x)))  #Remove ()


# 儲存CSV檔案
data.to_csv('D:\\MLforSchool\\data\\16qam_valid\\random_feature_forVaild.csv', index=False)

print(data)


