import pandas as pd
import numpy as np
import re



qam ='QPSK'
column = 31  #根據測試資料的列數更改
snr = 17
for snr in range(4):
    if snr == 0:
        channel = '1to28_fortrain' 
    if snr == 1:
        channel = '29to36_fortest'
    if snr == 2:
        channel = '1to8_fortrain'
    if snr == 3:
        channel = '9to19_fortest'
# 16point for h0 or h1
    var = 0.1586
    point_h0_csv = pd.read_csv(f'D:\\MLforSchool\\data\\constellations\\{qam}_for_0\\{qam}_4_15.csv')
    point_h1_csv = pd.read_csv(f'D:\\MLforSchool\\data\\constellations\\{qam}_for_1\\{qam}_4_15.csv')
    # channel_feature_csv = pd.read_csv(f'D:\\MLforSchool\\data\\256qam_for_channel\\0125_lab1_tu6\\lab1_snr16_256qamUi1.csv')
    # channel_feature_csv = pd.read_csv(f'D:\\MLforSchool\\data\\{qam}qam_for_channel\\{qam}qam_{ChannelFeatureData}\\lab2_256qamUi2_cr10_snr17_4000test.csv')
    # perfectH_square_2var_csv = pd.read_csv(f'D:\\MLforSchool\\data\\256qam_for_channel\\0125_lab1_tu6\\squaredH_divided_by_2var\\lab1_snr16_256qamUi1_coderate10_squaredH_divided_by_2var_real.csv')
    channel_feature_csv = pd.read_csv(f'D:\\Desktop\\data\\check_BER\\0307yuan_QPSK_8dB_cr4_TU6\\Y_BER_8p7_{channel}.csv')
    perfectH_csv = pd.read_csv(f'D:\\Desktop\\data\\check_BER\\0307yuan_QPSK_8dB_cr4_TU6\\complex_data_H8p7_perfect_{channel}.csv')
    def change_all_positive(x):
        return complex(abs(x.real), abs(x.imag))
    def change_i_to_j(x):
        return complex(x.replace('i', 'j'))
    point_h0_csv.replace('i', 'j', regex=True, inplace=True)
    point_h1_csv.replace('i', 'j', regex=True, inplace=True)

    channel_feature = channel_feature_csv.iloc[0:, 1:].applymap(change_i_to_j)
    perfectH = perfectH_csv.iloc[0:, 1:].applymap(change_i_to_j)
    print(channel_feature)
    print(perfectH)
    # 將所有複數取絕對值(象限壓縮)
    perfectH_square_2var_csv = (abs(perfectH))**2/var
    transform_to_positive = channel_feature.applymap(change_all_positive)

    def cal_distance(a, b):
        return abs(a - b) ** 2

    # {
    #   0: [ans, ans, ans...],
    #   1: [ans, ans, ans...],
    #    ...
    # }
    dict_for_bit_ans = {}

    for j, row_rf in transform_to_positive.iterrows():
        row_H =  perfectH_square_2var_csv.iloc[j]
        list_rf = row_rf.values.tolist()
        list_H = row_H.values.tolist()

        for k,random_feature_item in enumerate(list_rf):

            for i, row_h0 in point_h0_csv.iterrows():
                row_h1 = point_h1_csv.iloc[i]
                list_h0 = row_h0.values.tolist()
                list_h1 = row_h1.values.tolist()
                list_h0.pop(0)
                list_h1.pop(0)

                def cal_distance_of_random_feature_item(item):
                    return cal_distance(complex(item), complex(random_feature_item))
                def cal_min_distance_of_random_feature_item(array):
                    return min(list(map(cal_distance_of_random_feature_item, array)))
                    
                min_h0 = cal_min_distance_of_random_feature_item(list_h0)
                min_h1 = cal_min_distance_of_random_feature_item(list_h1)
                llr = (min_h1 - min_h0)*list_H[k]
                
                if llr >20:
                    llr = 20
                elif llr < -20:
                    llr = -20
                else:
                    pass
                
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
        csv.T.to_csv(f'D:\\Desktop\\data\\check_BER\\0307yuan_QPSK_8dB_cr4_TU6\\ans_positive_MaxLopllr_{channel}_b{key}.csv')








