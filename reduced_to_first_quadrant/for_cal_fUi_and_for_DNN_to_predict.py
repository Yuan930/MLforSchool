import pandas as pd
import re

data = 'test'  #train valid test

# 假設你有一個 DataFrame，包含複數數據
feature_in_channel_csv = pd.read_csv(f'D:\\MLforSchool\\data\\16qam_for_channel\\16qam_train\\lab4_16qamUi_coderate10_snr8_train.csv')

# 將複數變為絕對值的函數
def make_positive(x):
    return complex(abs(x.real), abs(x.imag))
# 由於python的複數是j，所以要把i變為j
def change_i_to_j(x):
    return complex(x.replace('i', 'j'))
# 去除括號
def remove_parentheses(x):
    return re.sub(r'[()]', '', str(x))
# 實虛分開
def split_real_and_imag(x):
    return f'[{x.real},{x.imag}]'
feature_in_channel_csv = feature_in_channel_csv.iloc[0:, 1:].applymap(change_i_to_j)
# 將所有複數取絕對值(象限壓縮)
transform_to_positive = feature_in_channel_csv.applymap(make_positive)
# 去除括號
remove_parentheses_transform_to_positive = transform_to_positive.applymap(remove_parentheses)

remove_parentheses_transform_to_positive.to_csv(f'D:\\MLforSchool\\data\\16qam_for_channel\\16qam_train\\lab4_16qamUi_coderate10_snr8_train_positive.csv')