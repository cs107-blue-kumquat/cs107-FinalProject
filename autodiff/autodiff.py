import numpy as np
import re

class Variable():
    def __init__(self, var, der = 1):
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
       return self.__add__(other)


    def __rmul__(self, other):
       return self.__mul__(other)


    def __sub__(self, other):
       return self.__add__(-other)


    def __rsub__(self, other):
       return (-self).__add__(other)


    def __truediv__(self, other):
        try:
            new_div = self.var / other.var
            new_der = (self.der * other.var - other.der * self.var) / (other.var)**2
            return Variable(new_div, new_der)
        except AttributeError:
            new_div = self.var / other
            new_der = self.der / other
            return Variable(new_div, new_der)


    def __neg__(self):
        return Variable(-self.var, -self.der)


    def __rtruediv__(self, other):
        try:
           new_div = other.var / self.var
           new_der = (other.der * self.var - other.var * self.der)/(self.var**2)
           # new_der = (other.der * self.var - self.der * other.var) / (self.var)**2
           return Variable(new_div, new_der)
        except AttributeError:
            new_div = other / self.var
            # base has different derivative, not sure of logic
            new_der = other * (self.var**(-2)) * self.der
            return Variable(new_div, new_der)


    def __lt__(self, other):
        try:
            return self.var < other.var
        except AttributeError:
            return self.var < other


    def __gt__(self, other):
        try:
            return self.var > other.var
        except AttributeError:
            return self.var > other


    def __le__(self, other):
        try:
            return self.var <= other.var
        except AttributeError:
            return self.var <= other


    def __ge__(self, other):
        try:
            return self.var >= other.var
        except AttributeError:
            return self.var >= other


    def __eq__(self, other):
        try:
            return self.var == other.var
        except:
            raise TypeError('Input is not comparable.')


    def __ne__(self, other):
        return not self.__eq__(other)


    def __abs__(self):
        return Variable(abs(self.var), abs(self.der))


    def __pow__(self, other):
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
        try:
            new_val = other ** self.var
        except:
            raise ValueError("{} must be a number.".format(other))
        new_der = other**self.var * np.log(other)
        return Variable(new_val, new_der)

        
    @staticmethod
    def log(var):
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
        try:
            new_val = np.arctan(var.var)
            new_der = var.der * 1 / (1 + np.power(var.var, 2))

            return Variable(new_val, new_der)

        except AttributeError:
            return np.arctan(var)


    @staticmethod
    def sinh(var):
        try:
            new_val = np.sinh(var.var)
            new_der = var.der * np.cosh(var.var)
            return Variable(new_val, new_der)

        except AttributeError:
            return np.sinh(var)


    @staticmethod
    def cosh(var):
        try:
            new_val = np.cosh(var.var)
            new_der = var.der * np.sinh(var.var)

            return Variable(new_val, new_der)

        except AttributeError:
            return np.cosh(var)


    @staticmethod
    def tanh(var):
        try:
            new_val = np.tanh(var.var)
            new_der = var.der * 1 / np.power(np.cosh(var.var), 2)
            return Variable(new_val, new_der)
        except AttributeError:
            return np.tanh(var)


    @staticmethod
    def logistic(var):
        try:
            logistic_var = 1 / (1 + np.exp(-var.var))
            logistic_der = logistic_var * (1-logistic_var) * var.der
            return Variable(logistic_var, logistic_der)
        except:
            raise TypeError(f"Input {var} not valid.")


class SimpleAutoDiff: 
    def __init__(self, dict_val, list_funct):
        for func in list_funct:
            if not isinstance(func, str):
                raise TypeError('Invalid function input.')

        for key, val in enumerate(dict_val):
            exec(val + "= Variable(dict_val[val])")
            
        static_elem_funct = ['log', 'sqrt', 'exp', 'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh', 'logistic']
        

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
