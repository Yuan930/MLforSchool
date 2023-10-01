import re

def flatten(l):
     return [item for sublist in l for item in sublist]

def remove_parentheses(x):
    return re.sub(r'[]', '', str(x))