import numpy as np


class elementary_function():
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
           return elementary_function(new_add, new_der)
        except: 
            if isinstance(other, int) or isinstance(other, float):
                # other is not a variable and the addition could complete if it is a real number
                return elementary_function(self.var + other, self.der)
            else:
                raise TypeError("Input is not a real number.")


    def __mul__(self, other):
        try:
            new_mul = other.var * self.var
            new_der = self.der * other.var + other.der * self.var
            return elementary_function(new_mul, new_der)
        except:
            if isinstance(other, int) or isinstance(other, float):
                # other is not a variable and the multiplication could complete if it is a real number
                new_mul = other * self.var
                new_der = self.der * other
                return elementary_function(new_mul, new_der)
            else:
                raise TypeError("Input is not a real number.")
        

    def __radd__(self, other):
       return self.__add__(other)


    def __rmul__(self, other):
       return self.__mul__(seother)


    def __sub__(self, other):
       return self.__add__(-other)


    def __rsub__(self, other):
       return (-self).__sub__


    def __truediv__(self, other):
        try:
            new_div = self.var / other.var
            new_der = (self.der * other.var - other.der * self.var) / (other.var)**2
            return elementary_function(new_div, new_der)
        except AttributeError:
            new_div = self.var / other
            new_der = self.der / other
            return elementary_function(new_div, new_der)


    def __neg__(self):
        return elementary_function(-self.var, -self.der)


    def __rtruediv__(self, other):
        try:
           new_div = other.var / self.var
           new_der = (other.der * self.var - self.der * other.var) / (self.var)**2
           return elementary_function(new_div, new_der)
        except AttributeError:
            new_div = self.var / other
            # base has different derivative, not sure of logic
            new_der = self.der / other
            return elementary_function(new_div, new_der)


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
        return elementary_function(abs(self.var), abs(self.der))


    def __pow__(self, other):
        try:
            new_val = self.var ** other.var
            new_der = self.var ** other.var * (np.log(self.var) * other.der / self.der + other.var / self.var)
            return elementary_function(new_val, new_der)
        except:
            if isinstance(other, int):
                new_val = self.var ** other
                new_der = other * self.var ** (other - 1) * self.der
                return elementary_function(new_val, new_der)
            else:
                raise TypeError(f"Exponent {other} is not valid.")


    def __rpow__(self, other):
        try:
            new_val = other ** self.var
        except:
            raise ValueError("{} must be a number.".format(other))
        new_der = other**self.var * np.log(other)
        return elementary_function(new_val, new_der)

        
    @staticmethod
    def log(var):
        try:
            if var.var <= 0:
                raise ValueError('Input needs to be greater than 0.')
        except :
            raise TypeError(f"Input not valid.")
        log_var = np.log(var.var)
        log_der = (1. / var.var) * var.der
        return elementary_function(log_var, log_der)


    @staticmethod
    def sqrt(var):
      if var < 0:
        raise ValueError("square root must take positive number")
      else:
        try:
          return elementary_function(var.var**(1/2), (1/2)*var.var**(-1/2))
        except AttributeError:
          return elementary_function(var**(1/2), (1/2)*var**(-1/2))



    @staticmethod
    def exp(self):

      try:
        new_val = np.exp(var.var)
        new_der = np.exp(var.var) * var.der
        return Variable(new_val, new_der)

      except AttributeError:
        return np.exp(var)


    @staticmethod
    def sin(self):
      try:
        new_val = np.sin(var.var)
        new_der = var.der * np.cos(var.var)
        return Variable(new_val, new_der)

      except AttributeError:
        return np.sin(var)


    @staticmethod
    def cos(self):
        try:
          new_val = np.cos(self.var)
          new_der = self.der * -np.sin(self.var)
          return elementary_function(new_val, new_der)

        except AttributeError:
            return np.cos(var)
    
    
    @staticmethod
    def tan(self):
        new_val = np.tan(self.var)
        new_der = variable.der * 1 / np.power(np.cos(variable.var), 2)
        return elementary_function(new_var, new_der)

    @staticmethod
    def arcsin(self):
        try:
            if self.var>1 or self.var <-1:
               raise ValueError('Please input -1 <= x <=1')
            else:
                new_var = np.arcsin(self.var)
                new_der = 1 / np.sqrt(1 - (self.var ** 2))
            return elementary_function(new_var, new_der) 
        except AttributeError:
            return np.arcsin(var)


    @staticmethod
    def arccos(self):
        try:
            if self.var>1 or self.var <-1:
                raise ValueError('Please input -1 <= x <=1')

            else:
                new_var = np.arcsin(self.var)
                new_der = -1 / np.sqrt(1 - (self.var ** 2))
            return elementary_function(new_var, new_der)
        except AttributeError:
            return np.arccos(var)


    @staticmethod
    def arctan(self):
        try:
            new_var = np.arctan(self.var)
            new_der = self.der * 1 / (1 + np.power(self.var, 2))

            return elementary_function(new_var, new_der)

        except AttributeError:
            return np.arctan(var)


    @staticmethod
    def sinh(self):
        try:
            new_var = np.sinh(self.var)
            new_der = self.der * np.cosh(self.var)

            return elementary_function(new_var, new_der)

        except AttributeError:
            return np.sinh(car)


    @staticmethod
    def cosh(self):
        try:
            new_var = np.cosh(self.var)
            new_der = self.der * np.sinh(self.var)

            return elementary_function(new_var, new_der)

        except AttributeError:
            return np.cosh(var)


    @staticmethod
    def tanh(self):
        try:
            new_var = np.tanh(self.var)
            new_der = self.der * 1 / np.power(np.cosh(self.var), 2)

            tanh = elementary_function(new_var, new_der)
            return tanh

        except AttributeError:
            return np.tanh(var)


    class SimpleAutoDiff: 
        def __init__(self, dict_val, list_funct):

        #     self.functions=np.array(len(dict_val))
        #     self.variables=np.array(len(list_funct))

            for key,val in enumerate(dict_val):
                str1= 'temp_'+val
                exec("%s = %d" % (str1,dict_val[val]))

            for key,val in enumerate(dict_val):
                str2= val
                exec(str2+"= elementary_function(dict_val[val])")
                print(str2+"= elementary_function(dict_val[val])")
            static_elem_funct = ['log', 'sqrt', 'exp', 'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh']

            n = len(list_funct)
            self.n = n


            #code to parse the strings in the list of functions
            global parsed_funct
            parsed_funct=[]
            len_list_funct = len(list_funct)
            #loop through all functions in argument list
            for funct in range(0,len_list_funct):
            #the eval function evaluates the “String” like a python expression and returns the result as an integer.
                parsed_funct.append(eval(list_funct[funct]))
            self.functions = parsed_funct


    def __repr__ (self):
        #include formated print statements here
        ret_string = ""
        return ret_string
    def __str__(self):
        output = '---AutoDifferentiation---\n'
        added_output=' '
        for i in range(0,self.n):
            added_output +='Function'+str(i+1)+' Value:'+str(parsed_funct[i].var)+' and ''Function'+str(i+1)+' Gradient:'+str(parsed_funct[i].der)+'\n'
        return output+added_output
