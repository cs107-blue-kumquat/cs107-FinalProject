import pytest
from autodiff.autodiff import *

def test_init():
	dict_val = {'x': 1,'y': 2}
	list_functs = ['x * 8', 'x*y']
	auto_diff_test = Reverse(dict_val, list_functs)

	dict_val = {'x': 1}
	list_functs = ['x * 8', 2]
	with pytest.raises(TypeError):	
		auto_diff_test = Reverse(dict_val, list_functs)


def test_elem_func():
	dict_val = {'x': 0.2}
	list_functs = ['log(x)', 'sqrt(x)', 'exp(x)',
				   'sin(x)', 'cos(x)', 'tan(x)', 
				   'arcsin(x)', 'arccos(x)', 'arctan(x)',
				   'sinh(x)', 'cosh(x)', 'tanh(x)',
				   'sigmoid(x)']
	auto_diff_test = Reverse(dict_val, list_functs)


def test_elem_func_plus():
	dict_val = {'x': 0.2, 'y':0.5}
	list_functs = ['log(x+y)', 'sqrt(x+y)', 'exp(x+y)',
				   'sin(x+y)', 'cos(x+y)', 'tan(x+y)', 
				   'arcsin(x+y)', 'arccos(x+y)', 'arctan(x+y)',
				   'sinh(x+y)', 'cosh(x+y)', 'tanh(x+y)',
				   'sigmoid(x+y)']
	auto_diff_test = Reverse(dict_val, list_functs)


def test_elem_func_times():
	dict_val = {'x': 0.2, 'y':0.5}
	list_functs = ['log(x*y)', 'sqrt(x*y)', 'exp(x*y)',
				   'sin(x*y)', 'cos(x*y)', 'tan(x*y)', 
				   'arcsin(x*y)', 'arccos(x*y)', 'arctan(x*y)',
				   'sinh(x*y)', 'cosh(x*y)', 'tanh(x*y)',
				   'sigmoid(x*y)']
	auto_diff_test = Reverse(dict_val, list_functs)


def test_repr():
	dict_val = {'x': 1}
	list_functs = ['x * 8']
	auto_diff_test = Reverse(dict_val, list_functs)

	temp = "---Reverse Differentiation---\n" \
			"Function 1: \nExpression = x * 8\nValue = 8.0\nGradient = [8]\n\n"
	assert repr(auto_diff_test) == temp


def test_str():
	dict_val = {'x': 1}
	list_functs = ['x * 8']
	auto_diff_test = Reverse(dict_val, list_functs)

	temp = "---Reverse Differentiation---\n" \
			"Function 1: \nExpression = x * 8\nValue = 8.0\nGradient = [8]\n\n"
	assert str(auto_diff_test) == temp
