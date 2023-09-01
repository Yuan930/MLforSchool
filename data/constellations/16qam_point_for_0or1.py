import pandas as pd
import re


# 16point
data = pd.read_csv('E:\\Huang_ATSC\\MLForSchool\\data\\constellations\\16qam_point.csv')
all_expanded_data = []

for i, row in data.iterrows():
    if i >= 1:
        complex_numbers = [complex(row[str(j)]) for j in range(1, 5)]
        
        # w_tb = [w, -w.conjugate(), w.conjugate(), -w]
        expanded_complex_numbers = [
            complex_numbers[0],                     # 第一象限
            complex_numbers[1],     
            complex_numbers[2],     
            complex_numbers[3]      
        ] + [
            -complex_numbers[0].conjugate(),        # 第二象限               
            -complex_numbers[1].conjugate(),
            -complex_numbers[2].conjugate(),
            -complex_numbers[3].conjugate()
        ] + [
            complex_numbers[0].conjugate(),         # 第四象限  
            complex_numbers[1].conjugate(),
            complex_numbers[2].conjugate(),
            complex_numbers[3].conjugate()
        ] + [
            -complex_numbers[0],                    # 第三象限  
            -complex_numbers[1],
            -complex_numbers[2],
            -complex_numbers[3]
        ]
        
        #Remove ()
        expanded_complex_numbers = [re.sub(r'[()]', '', str(num)) for num in expanded_complex_numbers]
        
        #print(expanded_complex_numbers)
        all_expanded_data.append(expanded_complex_numbers)

column_labels = [f'{j+1}' for j in range(len(expanded_complex_numbers))]

# Create a DataFrame from the list of expanded_complex_numbers
expanded_data_df = pd.DataFrame(all_expanded_data, columns=column_labels)

aa = expanded_data_df.to_csv('E:\\Huang_ATSC\\MLForSchool\\data\\constellations\\16qam_16point.csv')





# import pandas as pd
# import re
# # 16point
# data = pd.read_csv('E:\\Huang_ATSC\\Yuan_ML\\train_data\\16qam_point.csv')

# complex_data = []
# for i, row in data.iterrows():
#         if i >= 1:
#             index = int(row['id'].split('_')[0])
#             # 將複數儲存在字典中，使用 i 作為鍵
#             complex_data[index] = [complex(row['1']), complex(row['2']), complex(row['3']), complex(row['4'])]