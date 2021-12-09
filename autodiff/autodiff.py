import numpy as np
import re

class Variable():
    def __init__(self, var, der = 1):
        """
        Initialize a Variable object with follow attributes:
        var: attribute representing evaluated value 
        der: attribute representing evaluated derivative/gradient
        
        Derivative is 1 by default if not specified.
        """
        if isinstance(var, int) or isinstance(var, float):
            self.var = var
            self.der = der
        else:
            raise TypeError("Input is not a real number.")


    def __str__(self):
        return f"value = {self.var}, derivative = {self.der}"


    def __repr__(self):
        return f"value = {self.var}, derivative = {self.der}"


    def __add__(self, other):
        """
        dunder method for adding an Variable object.
        
        :param self: Variable object.
        :param other: Variable object, or int/float, to be added to self.var.
        :return: Variable object with value and derivative of the sum of self and other.
        """
        try:
           new_add = self.var + other.var
           new_der = self.der + other.der
           return Variable(new_add, new_der)
        except: 
            if isinstance(other, int) or isinstance(other, float):
                # other is not a variable and the addition could complete if it is a real number
                return Variable(self.var + other, self.der)
            else:
                raise TypeError("Input is not a real number.")


    def __mul__(self, other):
        """
        dunder method for multiplying an Variable object or constant.
        
        :param self: Variable object.
        :param other: Variable object, or int/float, to be multiplied to self.var.
        :return: Variable object with value and derivative of the product of self and other.
        """
        try:
            new_mul = other.var * self.var
            new_der = self.der * other.var + other.der * self.var
            return Variable(new_mul, new_der)
        except:
            if isinstance(other, int) or isinstance(other, float):
                # other is not a variable and the multiplication could complete if it is a real number
                new_mul = other * self.var
                new_der = self.der * other
                return Variable(new_mul, new_der)
            else:
                raise TypeError("Input is not a real number.")
        

    def __radd__(self, other):
        """
        dunder method for adding an Variable object and left object without __add__ mehtod.
        
        :param self: Variable object.
        :param other: an object that does not have an __add__ method or not implemented.
        :return: __add__ function call.
        """
        return self.__add__(other)


    def __rmul__(self, other):
        """
        dunder method for multiplying an Variable object and left object without __mul__ mehtod.
        
        :param self: Variable object.
        :param other: an object that does not have an __mul__ method or not implemented.
        :return: __mul__ function call.
        """
        return self.__mul__(other)


    def __sub__(self, other):
        """
        dunder method for subtracting an Variable object or constant.
        
        :param self: Variable object.
        :param other: Variable object, or int/float, to be subtracted from self.var.
        :return: __add__ function call that passes the other object with negative sign.
        """
        return self.__add__(-other)


    def __rsub__(self, other):
        """
        dunder method for subtracting an Variable object and left object without __sub__ mehtod.
        
        :param self: Variable object.
        :param other: an object that does not have an __sub__ method or not implemented.
        :return: __add__ function call that passes the self object with negative sign.
        """
        return (-self).__add__(other)


    def __truediv__(self, other):
        """
        dunder method for dividing an Variable object or constant.
        
        :param self: Variable object.
        :param other: Variable object, or int/float, to be divided from self.var.
        :return: Variable object with value and derivative of the fraction of self and other.
        """
        try:
            new_div = self.var / other.var
            new_der = (self.der * other.var - other.der * self.var) / other.var**2
            return Variable(new_div, new_der)
        except AttributeError:
            if isinstance(other, int) or isinstance(other, float):
                new_div = self.var / other
                new_der = self.der / other
                return Variable(new_div, new_der)
            else:
                raise TypeError(f"Input {other} is not valid.")


    def __neg__(self):
        """
        dunder method for taking the negative of an Variable object or constant.
        
        :param self: Variable object.
        :param other: Variable object, or int/float, to be taken the negative.
        :return: Variable object with negative value and derivative of itself.
        """
        return Variable(-self.var, -self.der)


    def __rtruediv__(self, other):
        """
        dunder method for subtracting an Variable object and left object without __truediv__ mehtod.
        
        :param self: Variable object.
        :param other: an object that does not have an __truediv__ method or not implemented.
        :return: Variable object with value and derivative of the fraction of self and other.
        """
        try:
            new_div = other.var / self.var
            new_der = (other.der * self.var - other.var * self.der) / self.var**2
            return Variable(new_div, new_der)
        except:
            if isinstance(other, int) or isinstance(other, float):
                new_div = other / self.var
                new_der = other * (self.var**(-2)) * self.der
                return Variable(new_div, new_der)
            else:
                raise TypeError(f"Input {other} is not valid.")


    def __lt__(self, other):
        '''
        dunder method for less than comparator.

        :param self: Variable object.
        :param other: Variable object, or int/float, to be compared with.
        :return: True if self is less than other, otherwise False.
        '''
        try:
            return self.var < other.var
        except AttributeError:
            return self.var < other


    def __gt__(self, other):
        '''
        dunder method for greater than comparator.

        :param self: Variable object.
        :param other: Variable object, or int/float, to be compared with.
        :return: True if self is greater than other, otherwise False.
        '''
        try:
            return self.var > other.var
        except AttributeError:
            return self.var > other


    def __le__(self, other):
        '''
        dunder method for less than or equal to comparator.

        :param self: Variable object.
        :param other: Variable object, or int/float, to be compared with.
        :return: True if self is less than or equal to other, otherwise False.
        '''
        try:
            return self.var <= other.var
        except AttributeError:
            return self.var <= other


    def __ge__(self, other):
        '''
        dunder method for greater than or equal to comparator.

        :param self: Variable object.
        :param other: Variable object, or int/float, to be compared with.
        :return: True if self is greater than or equal to other, otherwise False.
        '''
        try:
            return self.var >= other.var
        except AttributeError:
            return self.var >= other


    def __eq__(self, other):
        '''
        dunder method for equality comparator.

        :param self: Variable object.
        :param other: Variable object, or int/float, to be compared with.
        :return: True if self is equal to other, otherwise False.
        '''
        try:
            return self.var == other.var
        except:
            raise TypeError('Input is not comparable.')


    def __ne__(self, other):
        '''
        dunder method for inequality comparator.

        :param self: Variable object.
        :param other: Variable object, or int/float, to be compared with.
        :return: negation of __eq__ function call.
        '''
        return not self.__eq__(other)


    def __abs__(self):
        '''
        dunder method for absolute value.

        :param self: Variable object.
        :return: Variable object with value and derivative of the absolute value of self.
        '''
        return Variable(abs(self.var), abs(self.der))


    def __pow__(self, other):
        """
        dunder method for taking Variable object to the value of other object's power.
        
        :param self: Variable object.
        :param other: Variable object, or int/float, to the power of.
        :return: Variable object with value and derivative of the power of self with power of other's value.
        """
        try:
            new_val = self.var ** other.var
            new_der = self.var ** other.var * (np.log(self.var) * other.der / self.der + other.var / self.var)
            return Variable(new_val, new_der)
        except:
            if isinstance(other, int):
                new_val = self.var ** other
                new_der = other * self.var ** (other - 1) * self.der
                return Variable(new_val, new_der)
            else:
                raise TypeError(f"Exponent {other} is not valid.")


    def __rpow__(self, other):
        """
        dunder method for taking the other object to the Variable object's power.
        
        :param self: Variable object.
        :param other: Variable object, or int/float, as the base.
        :return: Variable object with value and derivative of the power of other with power of self's value.
        """
        try:
            new_val = other ** self.var
        except:
            raise ValueError("{} must be a number.".format(other))
        new_der = other**self.var * np.log(other)
        return Variable(new_val, new_der)

        
    @staticmethod
    def log(var):
        """
        takes the log of the Variable object.
        
        :param var: Variable object.
        :return: Variable object with value and derivative of the log of var.
        """
        try:
            if var.var <= 0:
                raise ValueError('Input needs to be greater than 0.')
        except:
            raise TypeError(f"Input not valid.")
        log_var = np.log(var.var)
        log_der = (1. / var.var) * var.der
        return Variable(log_var, log_der)


    @staticmethod
    def sqrt(var):
        """
        takes the square root of the Variable object.
        
        :param var: Variable object.
        :return: Variable object with value and derivative of the square root of var.
        """
        if var < 0:
            raise ValueError("Square root only takes positive number in the current implementation.")
        else:
            try:
                sqrt_var = var.var**(1/2)
                sqrt_der = (1/2)*var.var**(-1/2)
            except:
                raise TypeError(f"Input is not an Variable object.")
        return Variable(sqrt_var, sqrt_der)


    @staticmethod
    def exp(var):
        """
        natural exponential function with the value of Variable object as the power.
        
        :param var: Variable object.
        :return: Variable object with value and derivative of natural exponential function with power var.
        """
        try:
            new_val = np.exp(var.var)
            new_der = np.exp(var.var) * var.der
            return Variable(new_val, new_der)
        except:
            if not isinstance(var, int) and not isinstance(var, float):
                raise TypeError(f"Input {var} is not valid.")
        
            return Variable(np.exp(var), np.exp(var))


    @staticmethod
    def sin(var):
        """
        calculates the sine of Variable object.
        
        :param var: Variable object.
        :return: Variable object with value and derivative of the sine of var.
        """
        try:
            new_val = np.sin(var.var)
            new_der = var.der * np.cos(var.var)
            return Variable(new_val, new_der)
        except:
            if not isinstance(var, int) and not isinstance(var, float):
                raise TypeError(f"Input {var} is not valid.")
        
            return Variable(np.sin(var), np.cos(var))


    @staticmethod
    def cos(var):
        """
        calculates the cosine of Variable object.
        
        :param var: Variable object.
        :return: Variable object with value and derivative of the cosine of var.
        """
        try:
            new_val = np.cos(var.var)
            new_der = var.der * -np.sin(var.var)
            return Variable(new_val, new_der)
        except:
            if not isinstance(var, int) and not isinstance(var, float):
                raise TypeError(f"Input {var} is not valid.")
        
            return Variable(np.cos(var), -np.sin(var))
    
    
    @staticmethod
    def tan(var):
        """
        calculates the tangent of Variable object.
        
        :param var: Variable object.
        :return: Variable object with value and derivative of the tangent of var.
        """
        try:
            new_val = np.tan(var.var)
            new_der = var.der * 1 / np.power(np.cos(var.var), 2)
            return Variable(new_val, new_der)
        except:
            if not isinstance(var, int) and not isinstance(var, float):
                raise TypeError(f"Input {var} is not valid.")
        
            return Variable(np.tan(var), 1/np.cos(var)**2)


    @staticmethod
    def arcsin(var):
        """
        calculates the arcsine of Variable object.
        
        :param var: Variable object.
        :return: Variable object with value and derivative of the arcsine of var.
        """
        try:
            if var.var > 1 or var.var < -1:
                raise ValueError('Please input -1 <= x <=1')
            else:
                new_val = np.arcsin(var.var)
                new_der = 1 / np.sqrt(1 - (var.var ** 2))
                return Variable(new_val, new_der)
        except:
            if not isinstance(var, int) and not isinstance(var, float):
                raise TypeError(f"Input {var} is not valid.")
            return Variable(np.arcsin(var), 1 / np.sqrt(1 - (var ** 2)))


    @staticmethod
    def arccos(var):
        """
        calculates the arccosine of Variable object.
        
        :param var: Variable object.
        :return: Variable object with value and derivative of the arccosine of var.
        """
        try:
            if isinstance(var, int) or isinstance(var, float):
                return np.arccos(var)

            if var.var > 1 or var.var < -1:
                raise ValueError('Please input -1 <= x <=1')
            else:
                new_val = np.arccos(var.var)
                new_der = -1 / np.sqrt(1 - (var.var ** 2))
            return Variable(new_val, new_der)
        except:
                raise TypeError(f"Input {var} is not valid.")


    @staticmethod
    def arctan(var):
        """
        calculates the arctangent of Variable object.
        
        :param var: Variable object.
        :return: Variable object with value and derivative of the arctangent of var.
        """
        try:
            new_val = np.arctan(var.var)
            new_der = var.der * 1 / (1 + np.power(var.var, 2))

            return Variable(new_val, new_der)

        except AttributeError:
            return np.arctan(var)


    @staticmethod
    def sinh(var):
        """
        calculates the hyperbolic sine of Variable object.
        
        :param var: Variable object.
        :return: Variable object with value and derivative of the hyperbolic sine of var.
        """
        try:
            new_val = np.sinh(var.var)
            new_der = var.der * np.cosh(var.var)
            return Variable(new_val, new_der)

        except AttributeError:
            return np.sinh(var)


    @staticmethod
    def cosh(var):
        """
        calculates the hyperbolic cosine of Variable object.
        
        :param var: Variable object.
        :return: Variable object with value and derivative of the hyperbolic cosine of var.
        """
        try:
            new_val = np.cosh(var.var)
            new_der = var.der * np.sinh(var.var)

            return Variable(new_val, new_der)

        except AttributeError:
            return np.cosh(var)


    @staticmethod
    def tanh(var):
        """
        calculates the hyperbolic tangent of Variable object.
        
        :param var: Variable object.
        :return: Variable object with value and derivative of the hyperbolic tangent of var.
        """
        try:
            new_val = np.tanh(var.var)
            new_der = var.der * 1 / np.power(np.cosh(var.var), 2)
            return Variable(new_val, new_der)
        except AttributeError:
            return np.tanh(var)


    def sigmoid(var):
        """
        calculates the sigmoid/logistic function of Variable object.
        
        :param var: Variable object.
        :return: Variable object with value and derivative of the sigmoid/logistic function of var.
        """
        try:
            logistic_var = 1 / (1 + np.exp(-var.var))
            logistic_der = logistic_var * (1-logistic_var) * var.der
            return Variable(logistic_var, logistic_der)
        except:
            raise TypeError(f"Input {var} not valid.")


class SimpleAutoDiff: 
    def __init__(self, dict_val, list_funct):
        """
        How To Use SimpleAutoDiff
        ---------------------------------------------------------------------------------------------------------------------
    
        Inputs:
        
        dict_val: a dictionary object
        dict_val represents the input which is a dictionary of variables and their values formatted like the examples below
        example1={'x':7,'y':2, 'z':3}
        example2 ={'x':4}
        You can input as many variables as needed so long as it is more than 1
        
        list_funct: a list object
        list_funct represents the input which is a list of functions input as strings
        example_a = ['x**2','cos(np.pi*y)', '8*z']
        example_b = ['sin(x)']
        
        An example of a sample return using dict_val example1 and list_funct example_a would be 
        
        ---AutoDifferentiation---
        Value: {'x': 7, 'y': 2, 'z': 3}

        Function 1: 
        Expression = x**2
        Value = 49
        Gradient = 14

        Function 2: 
        Expression = cos(np.pi*y)
        Value = 1.0
        Gradient = 7.694682774887159e-16

        Function 3: 
        Expression = 8*z
        Value = 24
        Gradient = 8
        ---------------------------------------------------------------------------------------------------------------------
        
        """
        for func in list_funct:
            if not isinstance(func, str):
                raise TypeError('Invalid function input.')

        for key, val in enumerate(dict_val):
            exec(val + "= Variable(dict_val[val])")
            
        static_elem_funct = ['log', 'sqrt', 'exp', 'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh', 'sigmoid']
        

        self.dict_val = dict_val
        self.list_funct = list_funct
        self.functions = []

        len_list_funct = len(list_funct)

        # loop through all functions in argument list
        for func in list_funct:
            for elem_funct in static_elem_funct:
                if elem_funct in func: # e.g. log is in log(x)
                    func = 'Variable.' + func
                    break
            self.functions.append(eval(func))

    
    def __repr__ (self):
        output = '---AutoDifferentiation---\n'
        added_output = ''
        added_output += f"Value: {self.dict_val}\n\n"
        for i in range(0, len(self.functions)):
            added_output += f"Function {i+1}: \nExpression = {self.list_funct[i]}\nValue = {str(self.functions[i].var)}\nGradient = {str(self.functions[i].der)}\n\n"

        return output+added_output

    def __str__(self):
        output = '---AutoDifferentiation---\n'
        added_output = ''
        added_output += f"Value: {self.dict_val}\n\n"
        for i in range(0, len(self.functions)):
            added_output += f"Function {i+1}: \nExpression = {self.list_funct[i]}\nValue = {str(self.functions[i].var)}\nGradient = {str(self.functions[i].der)}\n\n"

        return output + added_output
