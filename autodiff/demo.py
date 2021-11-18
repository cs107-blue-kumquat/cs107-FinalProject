import numpy as np
from autodiff import *

dict_val = {'x': 1}
list_functs = ['x * 8', 'x + 3', 'log(x)', 'cos(x)', 'sqrt(x)', 'sinh(x)', 'arctan(x)']
auto_diff_test = SimpleAutoDiff(dict_val, list_functs)
print(auto_diff_test)
