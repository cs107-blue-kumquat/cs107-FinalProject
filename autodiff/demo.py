import numpy as np
from autodiff import *

# Variable  class
print("Variable Class Demo")
x = Variable.sin(np.pi)
y = Variable(2)

print("value:",np.round(x.var), "derivative:", x.der)

# Forward Mode Demo
print("Forward Mode Demo")
dict_val = {'x': 1}
list_functs = ['x * 8', 'x + 3', 'log(x)', 'cos(x)', 'sqrt(x)', 'sinh(x)', 'arctan(x)']
auto_diff_test = SimpleAutoDiff(dict_val, list_functs)
print(auto_diff_test)

# Demo Reverse Mode
dict_val = {'x': 1, 'y': 2}
list_funct = ['x * y + exp(x * y)', 'x + 3 * y']
out = Reverse(dict_val, list_funct)
print(out)