import pandas as pd
import numpy as np
import re

ChannelFeatureData = 'train'  #train valid test
qam =16
column = 100#根據測試資料的列數更改
N_var = 0.146322744985711
snr = 8.35
# 16point for h0 or h1
point_h0_csv = pd.read_csv(f'D:\\MLforSchool\\data\\constellations\\{qam}qam_for_0\\{qam}qam_10_15.csv')
point_h1_csv = pd.read_csv(f'D:\\MLforSchool\\data\\constellations\\{qam}qam_for_1\\{qam}qam_10_15.csv')
# channel_feature_csv = pd.read_csv(f'D:\\MLforSchool\\data\\{qam}qam_for_channel\\{qam}qam_{ChannelFeatureData}\\lab1_256qamUi_coderate10_snr{snr}_40000{ChannelFeatureData}.csv')
# channel_feature_csv = pd.read_csv(f'D:\\MLforSchool\\data\\{qam}qam_for_channel\\1106_lab5_r_100\\snr835\\lab5_16qamUi2_coderate10_snr8.35.csv')
channel_feature_csv = pd.read_csv(f'D:\\MLforSchool\\data\\{qam}qam_for_channel\\{qam}qam_{ChannelFeatureData}\\lab5_16qamUi_coderate10_snr8_35_train.csv')
# 將複數變為絕對值的函數
# print(channel_feature_csv)
def change_all_positive(x):
    return complex(abs(x.real), abs(x.imag))
def change_i_to_j(x):
    return complex(x.replace('i', 'j'))


point_h0_csv.replace('i', 'j', regex=True, inplace=True)
point_h1_csv.replace('i', 'j', regex=True, inplace=True)

channel_feature_csv = channel_feature_csv.iloc[0:, 1:].applymap(change_i_to_j)
# print(channel_feature_csv)
# 將所有複數取絕對值(象限壓縮)

transform_to_positive = channel_feature_csv.applymap(change_all_positive)
# print(transform_to_positive)

def cal_distance(a, b):
    return abs(a - b) ** 2
# {
#   0: [ans, ans, ans...],
#   1: [ans, ans, ans...],
#    ...
# }
dict_for_bit_ans = {}

for j, row_rf in transform_to_positive.iterrows(): 
    list_rf = row_rf.values.tolist()

    for random_feature_item in list_rf:

        for i, row_h0 in point_h0_csv.iterrows():
            row_h1 = point_h1_csv.iloc[i]
            list_h0 = row_h0.values.tolist()
            list_h1 = row_h1.values.tolist()
            list_h0.pop(0)
            list_h1.pop(0)

            def cal_distance_of_random_feature_item(item):
                return np.exp((-1/N_var)*cal_distance(complex(random_feature_item), complex(item)))
            def cal_min_distance_of_random_feature_item(array):
                return sum(list(map(cal_distance_of_random_feature_item, array)))
                
            pbk0 = cal_min_distance_of_random_feature_item(list_h0)
            pbk1 = cal_min_distance_of_random_feature_item(list_h1)
            llr = np.log(pbk0) - np.log(pbk1)
            print(llr)
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
    # csv.T.to_csv(f'D:\\MLforSchool\\data\\{qam}qam_for_channel\\{qam}qam_{ChannelFeatureData}\\ans\\lab1_LogMap_snr{snr}_LLR_result_b{key}_40000.csv')
    # csv.T.to_csv(f'D:\\MLforSchool\\data\\{qam}qam_for_channel\\1106_lab5_r_100\\ans\\lab5_16qamUi2_LogMap_snr835_LLR_result_b{key}.csv')
    csv.T.to_csv(f'D:\\MLforSchool\\data\\{qam}qam_for_channel\\{qam}qam_{ChannelFeatureData}\\ans\\lab5_LogMap_snr835_LLR_result_b{key}.csv')




