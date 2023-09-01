# MLforSchool

操作步驟:

1. use 16qam_random_LLR\\FeatureForRealAndImag.py 生成random feature，已當成train與test的feature
2. use data\\answer.py to calculate train's answer or test's real answer
    若是要做test(train)，請在16qam_test(train)做， 估算出的answer會在cal_answer
3. 使用CNN訓練模型，把train的特徵和答案輸入訓練模型，最後用test的特徵輸出估算出的答案
4. 使用MSE(Mean-Square Error)算出估(算出的答案)和(正確答案)的誤差




(#若需要其他的調變星座點，可參考data\\constellations\\16qam_point_for_0or1.py 的生成方式，記得csv裡要加id和空行)