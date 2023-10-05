import pandas as pd
import numpy as np
bit = 3
a = pd.read_csv(f'D:\\MLforSchool\\data\\16qam_for_channel\\16qam_test\\ans\\result_b{bit}.csv')


def divide(l):
    return l / 0.0158



b = a.iloc[0:, 1:].applymap(divide)
print(b)


b.to_csv(f'D:\\MLforSchool\\data\\16qam_for_channel\\16qam_test\\ans\\MAX_LOG_LLR_result_b{bit}.csv')