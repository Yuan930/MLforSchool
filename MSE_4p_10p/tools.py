import re

def flatten(l):
     return [item for sublist in l for item in sublist]

def change_i_to_j(x):
    return complex(x.replace('i', 'j'))

def change_to_complex(x):
    return complex(x)

def remove_parentheses(x):
    return re.sub(r'[()]', '', str(x))

def change_all_positive(x):
    return remove_parentheses(str(complex(abs(x.real), abs(x.imag))))

def split_real_and_imag(comp):
    array = comp.replace('j', '').split('+')
    return [float(array[0]), float(array[1])]


def Extract_real_parts(complexitem):
    real = complexitem.applymap(lambda x: x.real)
    return real

def Extract_imaginary_parts(complexitem):
    imag = complexitem.applymap(lambda x: x.imag)
    return imag