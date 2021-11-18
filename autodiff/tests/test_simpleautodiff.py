import pytest
import numpy as np
from autodiff.autodiff import *

def test_init():
	dict_val = {'x': 1}
	list_functs = ['x * 8']
	auto_diff_test = SimpleAutoDiff(dict_val, list_functs)

	dict_val = {'x': 1}
	list_functs = ['x * 8', 2]
	with pytest.raises(TypeError):	
		auto_diff_test = SimpleAutoDiff(dict_val, list_functs)


def test_elem_func():
	dict_val = {'x': 0.2}
	list_functs = ['log(x)', 'sqrt(x)', 'exp(x)',
				   'sin(x)', 'cos(x)', 'tan(x)', 
				   'arcsin(x)', 'arccos(x)', 'arctan(x)',
				   'sinh(x)', 'cosh(x)', 'tanh(x)',
				   'sigmoid(x)']
	auto_diff_test = SimpleAutoDiff(dict_val, list_functs)
