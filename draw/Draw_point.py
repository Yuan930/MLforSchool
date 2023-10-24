import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as rp
from tools import change_i_to_j, change_all_positive

point16_h0_csv = pd.read_csv('D:\\MLforSchool\\data\\constellations\\16qam_for_0\\16qam_10_15.csv')
point16_h1_csv = pd.read_csv('D:\\MLforSchool\\data\\constellations\\16qam_for_1\\16qam_10_15.csv')
feature_csv = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_channel\\16qam_train\\lab4_16qamUi_coderate10_snr8_train_positive.csv')
# feature_csv = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_randomfeature\\16qam_test\\random_feature100_fortest_with1.4.csv')
feature2_csv = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_channel\\16qam_test\\check\\lab4_check.csv')


feature_complex = feature_csv.iloc[0:, 1:].applymap(change_i_to_j) #for channel U'
feature2_complex_data = feature2_csv.iloc[0:, 1:].applymap(change_i_to_j)
feature2_complex_positive = feature2_complex_data.applymap(change_all_positive)
# print(feature_complex)
# print(feature2_complex_positive)

# feature_complex = feature_csv[['feature1', 'feature2']] #for uniform U'
point16_h0_complex = point16_h0_csv.iloc[0:, 1:].applymap(change_i_to_j)
point16_h1_complex = point16_h1_csv.iloc[0:, 1:].applymap(change_i_to_j)

# 提取实部和虚部
point16_h0_real = point16_h0_complex.applymap(lambda x: x.real)
point16_h0_imag = point16_h0_complex.applymap(lambda x: x.imag)

point16_h1_real = point16_h1_complex.applymap(lambda x: x.real)
point16_h1_imag = point16_h1_complex.applymap(lambda x: x.imag)

feature_real = feature_complex.applymap(lambda x: x.real)
feature_imag = feature_complex.applymap(lambda x: x.imag)

feature2_real = feature2_complex_data.applymap(lambda x: x.real)
feature2_imag = feature2_complex_data.applymap(lambda x: x.imag)
# 繪製點圖
plt.figure(figsize=(5, 5))  # 設置圖形大小

plt.scatter(feature_real, feature_imag, label='8dB awgn channel point', marker='s',color='c', s=5) #for channel U'
plt.scatter(feature2_real, feature2_imag, label='llr have different positive and negative', marker='o', color='b', s=50)
# plt.scatter(feature_complex['feature1'], feature_complex['feature2'], label="16QAM uniform Ui'", marker='s', s=1) #for uniform U'
plt.scatter(point16_h0_real, point16_h0_imag, label='16QAM constellations', marker='o', color='r', s=10)
plt.scatter(point16_h1_real, point16_h1_imag, marker='o', color='r', s=10)


plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.title("Ui' Plot")

plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)  # x軸線
plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)  # y軸線

# 顯示label位置
plt.legend(loc='upper right')
# plt.legend(loc=2, bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0 )
# plt.subplots_adjust(right=0.7)
# plt.subplots_adjust(top=0.7)

# 設置xy軸範圍
plt.xlim(-0.2, 2.5)
plt.ylim(-0.2, 2.5)

# 显示图形
plt.show()