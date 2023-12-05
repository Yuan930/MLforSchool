import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as rp
from tools import change_i_to_j, change_all_positive, Extract_real_parts, Extract_imaginary_parts, remove_parentheses
qam = 256
point_h0_csv = pd.read_csv(f'D:\\MLforSchool\\data\\constellations\\{qam}qam_for_0\\{qam}qam_10_15.csv')
point_h1_csv = pd.read_csv(f'D:\\MLforSchool\\data\\constellations\\{qam}qam_for_1\\{qam}qam_10_15.csv')
feature_csv = pd.read_csv(f'D:\\MLforSchool\\data\\{qam}qam_for_channel\\{qam}qam_train\\lab1_256qamUi_coderate10_snr173_20000train_positive.csv')
feature2_csv = pd.read_csv(f'D:\\MLforSchool\\data\\{qam}qam_for_channel\\{qam}qam_test\\lab1_256qamUi_coderate10_snr173_4000test_positive.csv')

point_h0bit3_csv = pd.read_csv(f'D:\\MLforSchool\\data\\constellations\\{qam}qam_for_0\\{qam}qam_bit3.csv')
point_h1bit3_csv = pd.read_csv(f'D:\\MLforSchool\\data\\constellations\\{qam}qam_for_1\\{qam}qam_bit3.csv')
# feature_csv = pd.read_csv('D:\\MLforSchool\\data\\{qam}_for_randomfeature\\{qam}_test\\random_feature100_fortest_with1.4.csv')
# feature2_csv = pd.read_csv('D:\\MLforSchool\\data\\{qam}_for_channel\\{qam}_test\\check\\lab4_check.csv')


feature_complex = feature_csv.iloc[0:, 1:].applymap(change_i_to_j) #for channel U'
feature2_complex_data = feature2_csv.iloc[0:, 1:].applymap(change_i_to_j)
# feature2_complex_positive = feature2_complex_data.applymap(change_all_positive)
# feature2 = feature2_complex_positive.applymap(remove_parentheses)

# print(feature2_complex_positive)

# feature_complex = feature_csv[['feature1', 'feature2']] #for uniform U'
point_h0_complex = point_h0_csv.iloc[0:, 1:].applymap(change_i_to_j)
point_h1_complex = point_h1_csv.iloc[0:, 1:].applymap(change_i_to_j)
point_h0_bit3 = point_h0bit3_csv.iloc[0:, 1:].applymap(change_i_to_j)
point_h1_bit3 = point_h1bit3_csv.iloc[0:, 1:].applymap(change_i_to_j)


# 繪製點圖
plt.figure(figsize=(5, 5))  # 設置圖形大小


# plt.scatter(feature_complex['feature1'], feature_complex['feature2'], label="16QAM uniform Ui'", marker='s', s=1) #for uniform U'
plt.scatter(Extract_real_parts(point_h0_complex), Extract_imaginary_parts(point_h0_complex), label='256QAM constellations', marker='o', color='r', s=10)
plt.scatter(Extract_real_parts(point_h1_complex), Extract_imaginary_parts(point_h1_complex), marker='o', color='r', s=10)

plt.scatter(Extract_real_parts(point_h1_bit3), Extract_imaginary_parts(point_h1_bit3), label='bit3 for 1', marker='o', color='c', s=15)
plt.scatter(Extract_real_parts(point_h0_bit3), Extract_imaginary_parts(point_h0_bit3), label='bit3 for 0', marker='o',color='b', s=15) #for channel U'

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
plt.xlim(0, 2)
plt.ylim(0, 2)

# 显示图形
plt.show()