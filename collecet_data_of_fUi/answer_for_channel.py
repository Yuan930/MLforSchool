import pandas as pd
import numpy as np
import re
# 16point for h0 or h1
point16_h0_csv = pd.read_csv('D:\\MLforSchool\\data\\constellations\\16qam_for_0\\16qam_10_15.csv')
point16_h1_csv = pd.read_csv('D:\\MLforSchool\\data\\constellations\\16qam_for_1\\16qam_10_15.csv')
#random_feature = pd.read_csv('D:\\MLforSchool\\data\\16qam_test\\random_feature100_forTest.csv')
#random_feature = pd.read_csv('D:\\MLforSchool\\data\\16qam_train\\random_feature1000_forTrain.csv')
random_feature_csv = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_channel\\fU_for_test.csv')


point16_h0_csv.replace('i', 'j', regex=True, inplace=True)
point16_h1_csv.replace('i', 'j', regex=True, inplace=True)
random_feature_csv.replace('i', 'j', regex=True, inplace=True)


def cal_distance(a, b):
    return abs(a - b)

# {
#   0: [ans, ans, ans...],
#   1: [ans, ans, ans...],
#    ...
# }
dict_for_bit_ans = {}

for j, row_rf in random_feature_csv.iterrows():

    list_rf = row_rf.values.tolist()
    list_rf.pop(0)
    for random_feature_item in list_rf:
        for i, row_h0 in point16_h0_csv.iterrows():
            row_h1 = point16_h1_csv.iloc[i]
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
            fUi = min_h1 - min_h0
            if i not in dict_for_bit_ans:
                dict_for_bit_ans[i] = []
            
            dict_for_bit_ans[i].append(fUi)
            
print(dict_for_bit_ans)
# 整理成原本random_feature.csv格式
for key in dict_for_bit_ans.keys():
    result = {}
    index = 0
    for item in dict_for_bit_ans[key]:
        if index not in result:
            result[index] = []
        result[index].append(item)
        if (len(result[index]) >= 10):  #根據測試資料的行數更改
            index = index + 1

    csv = pd.DataFrame(result)                
    csv.T.to_csv(f'D:\\MLforSchool\\data\\16qam_for_channel\\result_{key}.csv')





# fUi = pd.DataFrame(results[i], columns='ans')
# fUi.to_csv(f'D:\\MLforSchool\\data\\16qam_for_channel\\result_i_{i}.csv', index=False)

        


        

