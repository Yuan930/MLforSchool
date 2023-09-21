import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

point16_h0_csv = pd.read_csv('D:\\MLforSchool\\data\\constellations\\16qam_for_0\\16qam_10_15.csv')
point16_h1_csv = pd.read_csv('D:\\MLforSchool\\data\\constellations\\16qam_for_1\\16qam_10_15.csv')
#feature_csv = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_channel\\U.csv')
feature_csv = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_randomfeature\\random_feature100_forTest.csv')



point16_h0_complex = point16_h0_csv.iloc[1:, 1:].applymap(lambda x: complex(x.replace('i', 'j')))
point16_h1_complex = point16_h1_csv.iloc[1:, 1:].applymap(lambda x: complex(x.replace('i', 'j')))
# feature_complex = feature_csv.iloc[1:, 1:].applymap(lambda x: complex(x.replace('i', 'j')))
feature_complex = feature_csv[['feature1', 'feature2']]

# 提取实部和虚部
point16_h0_real = point16_h0_complex.applymap(lambda x: x.real)
point16_h0_imag = point16_h0_complex.applymap(lambda x: x.imag)

point16_h1_real = point16_h1_complex.applymap(lambda x: x.real)
point16_h1_imag = point16_h1_complex.applymap(lambda x: x.imag)

# feature_real = feature_complex.applymap(lambda x: x.real)
# feature_imag = feature_complex.applymap(lambda x: x.imag)

# 绘制散点图
plt.figure(figsize=(10, 6))  # 可选：设置图形大小
plt.scatter(point16_h0_real, point16_h0_imag, label='16QAM for 0', marker='o', s=10)
plt.scatter(point16_h1_real, point16_h1_imag, label='16QAM for 1', marker='x', s=10)
# plt.scatter(feature_real, feature_imag, label='16QAM for channel', marker='s', s=1)
plt.scatter(feature_complex['feature1'], feature_complex['feature2'], label='16QAM for channel', marker='s', s=1)


plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.title('Complex Scatter Plot')
# 显式指定图例位置
plt.legend(loc='upper right')

# 设置 x 轴和 y 轴的范围
plt.xlim(-3, 3)
plt.ylim(-3, 3)

# 显示图形
plt.show()