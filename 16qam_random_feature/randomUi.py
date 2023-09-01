import pandas as pd
import numpy as np
import random
import re

# 生成隨機複數
def generate_complex_number():
    x_real = random.uniform(0, 3)
    y_imag = random.uniform(0, 3)
    return complex(x_real, y_imag)

# 設定要生成的複數數量
num_complex_numbers = 1000

complex_numbers = []

# 生成複數
for i in range(num_complex_numbers):
    complex_number = generate_complex_number()
    complex_numbers.append(complex_number)
    
# 將複數列表轉換為 DataFrame
data = pd.DataFrame(complex_numbers, columns=['complex'])
data['complex'] = data['complex'].apply(lambda x: re.sub(r'[()]', '', str(x)))

#print(data)

aa = data.to_csv('F:\\Huang_ATSC\\Yuan_ML\\train_data\\random_complex_numbers1000.csv')