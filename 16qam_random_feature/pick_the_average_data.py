import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tools import change_i_to_j

def change_all_positive(x):
    return complex(abs(complex(x).real), abs(complex(x).imag))

# 讀取數據集
qam = 256
test_feature_csv = pd.read_csv(f'D:\\MLforSchool\\data\\{qam}qam_for_channel\\1120_lab1_r_100\\snr17\\lab1_256qamUi1_coderate10_snr17.csv')
test_feature_i_to_j = test_feature_csv.iloc[0:, 1:].applymap(change_i_to_j)
test_feature_all_positive = test_feature_i_to_j.applymap(change_all_positive)
list_test_all_positive_complex = test_feature_all_positive.values.flatten()

# 計算複數與原點的距離
distances = np.abs(list_test_all_positive_complex)

# 定義權重，距離越近權重越小，這裡將權重取倒數，讓外圈的複數在抽樣中佔較多比例
weights = 1 / (100000 + distances)

# 將權重標準化為概率
weights /= np.sum(weights)

# 定義抽樣數量
sample_size = 20000

# 從數據集中根據權重進行抽樣
sampled_indices = np.random.choice(len(list_test_all_positive_complex), size=sample_size, replace=False, p=weights)
sampled_complex_numbers = list_test_all_positive_complex[sampled_indices]

# 將複數分離成實部和虛部
real_parts_sampled = np.real(sampled_complex_numbers)
imaginary_parts_sampled = np.imag(sampled_complex_numbers)

# 打印結果
plt.scatter(real_parts_sampled, imaginary_parts_sampled, alpha=0.5, s=5)
plt.title('Scatter Plot of Sampled Complex Numbers')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.show()


