from keras.models import Sequential
from keras.layers import *
import pandas as pd
import numpy as np
from pandas import DataFrame
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def transpose(list1):
    return [list(row) for row in zip(*list1)]

data_1 = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_randomfeature\\16qam_train\\ans\\train_10_15_10000_1.4.csv')
x_train = data_1[['feature1','feature2']].values

data_2 = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_randomfeature\\16qam_valid\\ans\\valid_10_15_100_1.4.csv')
x_valid = data_2[['feature1','feature2']].values

data_3 = pd.read_csv('D:\\MLforSchool\\data\\16qam_for_randomfeature\\16qam_test\\random_feature100_fortest_with1.4.csv')
x_test = data_3[['feature1','feature2']].values

# 準備標籤數據：將所有比特位一起作為標籤
y_train = data_1[['b0', 'b1', 'b2', 'b3']].values
y_valid = data_2[['b0', 'b1', 'b2', 'b3']].values
print(y_valid)
print('x_test',x_test)
# 建立神經網路模型
model = Sequential()

# 添加輸入層和隱藏層
model.add(Dense(15, input_dim=2, kernel_initializer='normal',activation='relu'))
model.add(Dense(15, input_dim=15,  kernel_initializer='normal',activation='relu'))
model.add(Dense(15, input_dim=15,  kernel_initializer='normal',activation='relu'))
model.add(Dense(15, input_dim=15,  kernel_initializer='normal',activation='relu'))
model.add(Dense(4, kernel_initializer='normal',activation='linear'))

# 編譯模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 訓練模型
epochs = 40
batch_size = 100
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs ,verbose=1,validation_data=(x_valid, y_valid))
# 使用模型進行預測
bb = model.predict(x_test)
print(bb)

# 將預測結果進行轉置並創建 DataFrame
#a = transpose(bb)

a = DataFrame(bb)

# 設置列名
#a.columns = [f'b{i}' for i in range(0, 4)]
a.columns = ['b0', 'b1', 'b2', 'b3']
#a.to_csv('D://MLforSchool//dnn_experiments//independent_ans//combine.csv')
