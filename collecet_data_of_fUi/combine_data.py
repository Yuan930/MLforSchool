import pandas as pd
import numpy as np
ChannelFeatureData = 'test'
dict = {}
def csv_to_list(ui, snr):
    # file_path = f'D:\\MLforSchool\\data\\256qam_for_channel\\0125_lab1_tu6\\lab1_snr{snr}_256qamUi{ui}_coderate10.csv'
    file_path = f'D:\\MLforSchool\\data\\256qam_for_channel\\0125_lab1_tu6\\squaredH_divided_by_2var\\lab1_snr{snr}_256qamUi{ui}_coderate10_squaredH_divided_by_2var_real.csv'
    return list(pd.read_csv(file_path).iloc[0:100, 1:21].values.flatten())

# Initialize the lists
s_lists = []

# Loop through UI values (1 and 2)
for ui in range(5, 9):
    # For each UI, loop through SNR values (17 to 21)
    s_ui = [csv_to_list(ui, snr) for snr in range(17, 22)]
    s_lists.append(s_ui)

# Unpack the lists for individual variables
s1, s2, s3, s4, s5 = s_lists[0]
s6, s7, s8, s9, s10 = s_lists[1]
s11, s12, s13, s14, s15 = s_lists[2]
s16, s17, s18, s19, s20 = s_lists[3]

# list_a = list(a.iloc[0:,1:].values.flatten())
# list_b = list(b.iloc[0:,1:].values.flatten())
# list_c = list(c.iloc[0:,1:].values.flatten())

# print("list_a",l。ist_a)
combine_list = list(item for sublist in zip(s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20) for item in sublist)
dict = combine_list

result = {}
index = 0
for item in dict:
    if index not in result:
        result[index] = []
    result[index].append(item)
    if (len(result[index]) >= 400):  #根據測試資料的列數更改
        index = index + 1
# print(result)

llr = pd.DataFrame(result)

llr.T.to_csv((f'D:\\MLforSchool\\data\\256qam_for_channel\\TU6_256qam_{ChannelFeatureData}\\lab1_TU6_cr10_snr17_to_21_Ui1_to_4_squaredH_divided_by_2var_40000{ChannelFeatureData}.csv'))
# llr.T.to_csv((f'D:\\MLforSchool\\data\\256qam_for_channel\\TU6_256qam_{ChannelFeatureData}\\lab1_TU6_cr10_snr17_to_21_Ui1_to_4_40000{ChannelFeatureData}.csv'))