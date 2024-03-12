import pandas as pd
import numpy as np
import re

ChannelFeatureData = 'test'  #train valid test
qam =16
column = 16200#根據測試資料的列數更改
N_var = 0.1463
snr = 17
# 16point for h0 or h1
point_h0_csv = pd.read_csv(f'D:\\MLforSchool\\data\\constellations\\{qam}qam_for_0\\{qam}qam_10_15.csv')
point_h1_csv = pd.read_csv(f'D:\\MLforSchool\\data\\constellations\\{qam}qam_for_1\\{qam}qam_10_15.csv')
channel_feature_csv = pd.read_csv(f'D:\\MLforSchool\\data\\16qam_for_channel\\1106_lab5_r_100\\snr835\\lab5_16qamUi2_coderate10_snr8.35.csv')
# channel_feature_csv = pd.read_csv(f'D:\\MLforSchool\\data\\{qam}qam_for_channel\\1120_lab1_r_100\\snr17\\lab1_256qamUi3_coderate10_snr17.csv')
# channel_feature_csv = pd.read_csv(f'D:\\MLforSchool\\data\\{qam}qam_for_channel\\{qam}qam_{ChannelFeatureData}\\lab2_256qamUi2_cr10_snr17_4000test.csv')
# 將複數變為絕對值的函數
# print(channel_feature_csv)
def change_all_positive(x):
    return complex(abs(x.real), abs(x.imag))
def change_i_to_j(x):
    return complex(x.replace('i', 'j'))


point_h0_csv.iloc[0:, 1:].replace('i', 'j', regex=True, inplace=True)
point_h1_csv.iloc[0:, 1:].replace('i', 'j', regex=True, inplace=True)

channel_feature_csv = channel_feature_csv.iloc[0:, 1:].applymap(change_i_to_j)
print(channel_feature_csv)
# 將所有複數取絕對值(象限壓縮)

# transform_to_positive = channel_feature_csv.applymap(change_all_positive)
# print(transform_to_positive)

def cal_distance(a, b):
    return abs(a - b) ** 2
# {
#   0: [ans, ans, ans...],
#   1: [ans, ans, ans...],
#    ...
# }
dict_for_bit_ans = {}

for j, row_rf in channel_feature_csv.iterrows(): 
    list_rf = row_rf.values.tolist()

    for random_feature_item in list_rf:
        

        for i, row_h0 in point_h0_csv.iterrows():
            row_h1 = point_h1_csv.iloc[i]
            list_h0 = row_h0.values.tolist()
            list_h1 = row_h1.values.tolist()
            list_h0.pop(0)
            list_h1.pop(0)
            
            def cal_distance_of_random_feature_item(item):
                return np.exp(((-1)/N_var)*cal_distance(complex(random_feature_item), complex(item)))
            def cal_min_distance_of_random_feature_item(array):
                return sum(list(map(cal_distance_of_random_feature_item, array)))
                
            pbk0 = cal_min_distance_of_random_feature_item(list_h0)
            pbk1 = cal_min_distance_of_random_feature_item(list_h1)
            llr = np.log(pbk0) - np.log(pbk1)
            # print(llr)
            if i not in dict_for_bit_ans:
                dict_for_bit_ans[i] = []
            
            dict_for_bit_ans[i].append(llr)
            
# 整理成原本random_feature.csv格式
for key in dict_for_bit_ans.keys():
    result = {}
    index = 0
    for item in dict_for_bit_ans[key]:
        if index not in result:
            result[index] = []
        result[index].append(item)
        if (len(result[index]) >= column):  #根據測試資料的列數更改
            index = index + 1
    # print(result)

    csv = pd.DataFrame(result)                
    csv.T.to_csv(f'D:\\MLforSchool\\data\\{qam}qam_for_channel\\1106_lab5_r_100\\ans\\16qamUi2_LogMap_snr835_LLR_result_b{key}.csv')
    # csv.T.to_csv(f'D:\\MLforSchool\\data\\{qam}qam_for_channel\\1120_lab1_r_100\\snr17\\ans\\lab1_256qamUi3_LogMap_snr17_LLR_result_b{key}.csv')
    # csv.T.to_csv(f'D:\\MLforSchool\\data\\{qam}qam_for_channel\\{qam}qam_{ChannelFeatureData}\\ans\\lab2_LogMap_snr17_LLR_result_b{key}_4000.csv')

dict = {}
a = pd.read_csv(f'D:\\MLforSchool\\data\\{qam}qam_for_channel\\1106_lab5_r_100\\ans\\16qamUi2_LogMap_snr835_LLR_result_b0.csv')
b = pd.read_csv(f'D:\\MLforSchool\\data\\{qam}qam_for_channel\\1106_lab5_r_100\\ans\\16qamUi2_LogMap_snr835_LLR_result_b1.csv')
c = pd.read_csv(f'D:\\MLforSchool\\data\\{qam}qam_for_channel\\1106_lab5_r_100\\ans\\16qamUi2_LogMap_snr835_LLR_result_b2.csv')
d = pd.read_csv(f'D:\\MLforSchool\\data\\{qam}qam_for_channel\\1106_lab5_r_100\\ans\\16qamUi2_LogMap_snr835_LLR_result_b3.csv')
list_a = list(a.iloc[0:,1:].values.flatten())
list_b = list(b.iloc[0:,1:].values.flatten())
list_c = list(c.iloc[0:,1:].values.flatten())
list_d = list(d.iloc[0:,1:].values.flatten())
# print("list_a",l。ist_a)
combine_list = list( item for pair in zip(list_a, list_b, list_c, list_d) for item in pair)
dict = combine_list

result = {}
index = 0
for item in dict:
    if index not in result:
        result[index] = []
    result[index].append(item)
    if (len(result[index]) >= 64800):  #根據測試資料的列數更改
        index = index + 1
# print(result)

llr = pd.DataFrame(result)

llr.T.to_csv((f'D:\\MLforSchool\\data\\{qam}qam_for_channel\\1106_lab5_r_100\\ans\\lab5_16qamUi2_LogMap_snr835_LLR_result.csv'))



