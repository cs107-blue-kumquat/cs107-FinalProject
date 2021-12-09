from autodiff import *

if __name__ == "__main__":
	dict_val = {"z":8, "b":16, "k":3, "g":5, "n":5}
	list_functs = ['-(z*2)+(0.5*b)','(b**2)+k','sin(4*k)-g','(n**3)+(z*2)','(n*k)','(b+7)*k']
	auto_diff_test = SimpleAutoDiff(dict_val, list_functs)
	print(auto_diff_test)


	dict_val = {'x': 1}
	list_functs = ['x * 8']
	auto_diff_test = SimpleAutoDiff(dict_val, list_functs)
	print(auto_diff_test) 
	temp = "Function 1: \nExpression = x * 8\nValue = 8\nGradient = [8.]\n\n"
	print(temp)
