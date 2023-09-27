import pandas as pd
import re
# 假設你有一個 DataFrame，包含複數數據
feature_in_channel_csv = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_channel\\16qam_test\\fU_for_test.csv')

# 定義將複數變為絕對值的函數
def make_positive(x):
    return complex(abs(x.real), abs(x.imag))
def change_i_to_j(x):
    return complex(x.replace('i', 'j'))
def remove_parentheses(x):
    return re.sub(r'[()]', '', str(x))
def split_real_and_imag(x):
    return pd.Series([x.real,x.imag])
feature_in_channel_csv = feature_in_channel_csv.iloc[0:, 1:].applymap(change_i_to_j)
# 將所有複數變為其絕對值
transform_to_positive = feature_in_channel_csv.applymap(make_positive)
# 去除括號
remove_parentheses_transform_to_positive = transform_to_positive.applymap(remove_parentheses)

# transform_to_positive.to_csv('D:\\MLforSchool\\data\\16qam_for_channel\\16qam_test\\fU_for_test_positive.csv')

split_transform_to_positive = transform_to_positive.applymap(split_real_and_imag)
split_transform_to_positive.to_csv('D:\\MLforSchool\\data\\16qam_for_channel\\16qam_test\\fU_for_test_split.csv')
print(split_transform_to_positive)

