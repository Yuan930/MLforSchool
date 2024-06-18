# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 09:52:42 2022

@author: oscar
"""

import matplotlib.pyplot as plt
import numpy as np
#plt.yscale("log")
#a = ['4+3','6+5','8+7','10+9','12+11','14+13']
import matplotlib.pyplot as plt

bit = ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']
# bit = ['b0', 'b1', 'b2', 'b3']
logmap_and_maxlog = [0.120, 0.141, 0.225, 0.147, 0.130, 0.085, 0.042, 0.017
]
dnn_fuc1 = [0.048, 0.053, 0.279, 0.442, 1.528, 0.790, 1.362, 0.635
]
dnn_fuc2 = [0.040, 0.044, 0.248, 0.220, 0.973, 0.883, 1.383, 0.389
]
dnn_fuc3 = [0.058, 0.105, 0.073, 0.136, 0.346, 0.195, 0.369, 0.424
]
dnn_fuc5 = [0.014, 0.010, 0.024, 0.014, 0.010, 0.023, 0.039, 0.029
]

plt.xlabel("bit")
plt.ylabel("MSE")

plt.yscale("log")  # 使用对数刻度

plt.plot(bit, logmap_and_maxlog, 'blue', marker='o', linewidth=2, label='MaxLog',linestyle='-.')
plt.plot(bit, dnn_fuc1, 'g', marker='^', label='DNN_lab1a',linestyle='-.')
plt.plot(bit, dnn_fuc2, 'm', marker='^', label='DNN_lab1b',linestyle='-.')
plt.plot(bit, dnn_fuc3, 'y', marker='^', label='DNN_lab2',linestyle='-.')
plt.plot(bit, dnn_fuc5, 'p', marker='^', label='DNN_lab3',linestyle='-.')
plt.yticks([1, 0.1, 0.01])
plt.legend(loc='upper left')
plt.show()
# lg = plt.legend(loc='upper right', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, mode='expanded')
# plt.savefig("pilot_個數不同對機器學習的影響_TU-6_CR1.png",dpi=300,format="png")
# plt.savefig("pilot_個數不同對機器學習的影響_TU-6_V68_BER.png",dpi=300,format="png",bbox_extra_artists=(lg,),bbox_inches='tight')
# bar_width = 0.1
# index = np.arange(len(bit))
# plt.xlabel("bit")
# plt.ylabel("MSE")
# plt.yscale("log")
# plt.bar(index, logmap_and_maxlog, width=bar_width, color='y', label='logmap_and_maxlog')
# plt.bar(index + bar_width, dnn_fuc1, width=bar_width, color='g', label='DNN_func1')
# plt.bar(index + 2 * bar_width, dnn_fuc2, width=bar_width, color='brown', label='DNN_func2')
# plt.bar(index + 3 * bar_width, dnn_fuc3, width=bar_width, color='m', label='DNN_func3')
# plt.bar(index + 4 * bar_width, dnn_fuc5, width=bar_width, color='b', label='DNN_func5')

# plt.xticks(index + 2 * bar_width, bit)
# plt.legend(loc='upper left')
# plt.show()
