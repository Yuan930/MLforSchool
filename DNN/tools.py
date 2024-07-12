import re
import numpy as np
def flatten(l):
     return [item for sublist in l for item in sublist]

def change_i_to_j(x):
    return complex(x.replace('i', 'j'))

def remove_parentheses(x):
    return re.sub(r'[()]', '', str(x))

def change_all_positive(x):
    return remove_parentheses(str(complex(abs(x.real), abs(x.imag))))

def split_real_and_imag_allpositive(comp):
    array = comp.replace('j', '').split('+')
    return [float(array[0]), float(array[1])]

def split_real_and_imag(complex_array):
    real_part = complex_array.real
    imag_part = complex_array.imag
    result = np.vstack((real_part, imag_part)).T.tolist()
    return result