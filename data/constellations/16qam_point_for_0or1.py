import pandas as pd
import re


# 16point
data = pd.read_csv('F:\\Huang_ATSC\\MLForSchool\\data\\16qam_point.csv')
all_expanded_data = []

for i, row in data.iterrows():
    if i >= 1:
        index = int(row['id'].split('_')[0])
        
        # Convert the string representation of complex numbers to actual complex numbers
        complex_numbers = [complex(row[str(j)]) for j in range(1, 5)]
        
        # Apply the formula w_tb = [w, -w.conjugate(), w.conjugate(), -w] to expand to all quadrants
        expanded_complex_numbers = [
            complex_numbers[0],                  # First quadrant
            complex_numbers[1],     # Second quadrant
            complex_numbers[2],      # Third quadrant
            complex_numbers[3]                  # Fourth quadrant
        ] + [
            -complex_numbers[0].conjugate(),                   # Repeat for the other points as well
            -complex_numbers[1].conjugate(),
            -complex_numbers[2].conjugate(),
            -complex_numbers[3].conjugate()
        ] + [
            complex_numbers[0].conjugate(),
            complex_numbers[1].conjugate(),
            complex_numbers[2].conjugate(),
            complex_numbers[3].conjugate()
        ] + [
            -complex_numbers[0],
            -complex_numbers[1],
            -complex_numbers[2],
            -complex_numbers[3]
        ]
        
        #Remove ()
        expanded_complex_numbers = [re.sub(r'[()]', '', str(num)) for num in expanded_complex_numbers]
        
        #print(expanded_complex_numbers)
        all_expanded_data.append(expanded_complex_numbers)

# Create column labels for the DataFrame
column_labels = [f'{j+1}' for j in range(len(expanded_complex_numbers))]

# Create a DataFrame from the list of expanded_complex_numbers
expanded_data_df = pd.DataFrame(all_expanded_data, columns=column_labels)

aa = expanded_data_df.to_csv('F:\\Huang_ATSC\\MLForSchool\\data\\16qam_16point.csv')
        




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