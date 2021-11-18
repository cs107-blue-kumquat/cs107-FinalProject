import pytest
import numpy as np
from autodiff import *


def test_init():
    x = elementary_function(2)
    assert x.var == 2
    assert x.der == 1
    with pytest.raises(TypeError):
        x = elementary_function("abc")


def test_add():
    # add two elementary_function objects
    x = elementary_function(1)
    y = x + elementary_function(2)
    assert y.var == 3
    assert y.der == 2

    # add a int/float to elementary_function object
    x = elementary_function(1)
    y = x + 2.
    assert y.var == 3
    assert y.der == 1
    
    # add an invalid type of input to elementary_function object
    with pytest.raises(TypeError):
        x = elementary_function(1)
        y = x + "a"


def test_neg():
    x = elementary_function(1)
    y = -x
    assert y.var == -1
    assert y.der == -1


def test_sub():
    # subtract two elementary_function objects
    x = elementary_function(1)
    y = x - elementary_function(2)
    assert y.var == -1
    assert y.der == 0

    # subtract a int/float to elementary_function object
    x = elementary_function(1)
    y = x - 2.
    assert y.var == -1
    assert y.der == 1
    
    # subtract an invalid type of input with elementary_function object
    with pytest.raises(TypeError):
        x = elementary_function(1)
        y = x - "a"


def test_mul():
    # multiply two elementary_function objects
    x = elementary_function(1)
    y = x * elementary_function(2)
    assert y.var == 2
    assert y.der == 3

    # multiply a int/float to elementary_function object
    x = elementary_function(1)
    y = x * 2.
    assert y.var == 2
    assert y.der == 2
    
    # multiply an invalid type of input with elementary_function object
    with pytest.raises(TypeError):
        x = elementary_function(1)
        y = x * "a"


def test_truediv():
    # divide two elementary_function objects
    x = elementary_function(1)
    y = x / elementary_function(2)
    assert y.var == 1/2
    assert y.der == 1/4

    # divide a int/float by elementary_function object
    x = elementary_function(1)
    y = x / 5.
    assert y.var == 1/5
    assert y.der == 1/5
    
    # divide elementary_function object by an invalid type of input 
    with pytest.raises(TypeError):
        x = elementary_function(1)
        y = x / "a"

    # divide elementary_function object by an invalid type of input 
    with pytest.raises(TypeError):
        x = elementary_function(1)
        y = "a" / x


def test_lt():
    x = elementary_function(1)
    y = elementary_function(2)
    assert (x < y) == True

    x = elementary_function(1)
    assert (x < 1) == False

    with pytest.raises(TypeError):
        x = elementary_function(1)
        check = "a" < x


def test_gt():
    x = elementary_function(1)
    y = elementary_function(2)
    assert (y > x) == True

    x = elementary_function(1)
    assert (x > 1) == False

    with pytest.raises(TypeError):
        x = elementary_function(1)
        check = "a" > x


def test_le():
    x = elementary_function(1)
    y = elementary_function(2)
    assert (x <= y) == True

    x = elementary_function(1)
    assert (x <= 1) == True

    with pytest.raises(TypeError):
        x = elementary_function(1)
        check = "a" <= x


def test_ge():
    x = elementary_function(1)
    y = elementary_function(2)
    assert (y >= x) == True

    x = elementary_function(1)
    assert (x >= 1) == True

    with pytest.raises(TypeError):
        x = elementary_function(1)
        check = "a" >= x

def test_eq():
    x = elementary_function(1)
    y = elementary_function(2)
    assert (y == x) == False

    with pytest.raises(TypeError):
        x = elementary_function(1)
        check = x == 1

def test_ne():
    x = elementary_function(1)
    y = elementary_function(2)
    assert (y != x) == True

    with pytest.raises(TypeError):
        x = elementary_function(1)
        check = x != 1


def test_abs():
    x = abs(elementary_function(1))
    assert x.var == 1
    assert x.der == 1

    y = abs(elementary_function(-2))
    assert y.var == 2
    assert y.der == 1


def test_pow():
    x = elementary_function(2)
    y = x ** 4
    assert y.var == 2 ** 4
    assert y.der == 4 * (2 ** 3)

    x = elementary_function(2)
    y = x ** -4
    assert y.var == 2 ** -4
    assert y.der == -4 * (2 ** -5)

    x = elementary_function(2)
    y = x ** x
    assert y.var == 2 ** 2
    assert y.der == 2 ** 2 * (np.log(2) * 1 / 1 + 2 / 2)

    x = elementary_function(2)
    y = x ** (x * 2)
    assert y.var == 2 ** (2 * 2)
    assert y.der == 2 ** (2 * 2) * (np.log(2) * 2 / 1 + (2 * 2) / 2)

    with pytest.raises(TypeError):
        x = elementary_function(1)
        check = x ** 1.2

    with pytest.raises(TypeError):
        x = elementary_function(1)
        check = x ** -1.2


def test_rpow():
    x = elementary_function(2)
    y = 4 ** x
    assert y.var == 4 ** 2
    assert y.der == 4 ** 2 * np.log(4)

    with pytest.raises(Exception):
        x = elementary_function(2)
        y = -4 ** x
        assert y.var == 4 ** 2
        assert y.der == 4 ** 2 * np.log(4)

    x = elementary_function(2)
    y = 4 ** (x * 2)
    assert y.var == 4 ** (2 * 2)
    assert y.der == 4 ** (2 * 2) * np.log(4)


def test_log():
    x = elementary_function(2)
    y = elementary_function.log(x)
    assert y.var == np.log(2)
    assert y.der == 1 / 2

    with pytest.raises(TypeError):
        x = elementary_function(-2)
        y = elementary_function.log(x)

    with pytest.raises(TypeError):
        y = elementary_function.log(2)
        assert y.var == np.log(2)
        assert y.der == 1 / 2


def test_sqrt():
    x = elementary_function(2)
    y = elementary_function.sqrt(x)
    assert y.var == np.sqrt(2)
    assert y.der == 1/2 * 2 ** (-1/2)

    with pytest.raises(ValueError):
        x = elementary_function(-2)
        y = elementary_function.sqrt(x)

    with pytest.raises(TypeError):
        y = elementary_function.sqrt("a")

    with pytest.raises(TypeError):
        y = elementary_function.sqrt(2)
        assert y.var == np.sqrt(2)
        assert y.der == 1/2 * 2 ** (-1/2)


def test_exp():
    x = elementary_function(2)
    y = elementary_function.exp(x)
    assert y.var == np.exp(2)
    assert y.der == np.exp(2)

    with pytest.raises(TypeError):
        y = elementary_function.exp("a")

    y = elementary_function.exp(2)
    assert y.var == np.exp(2)
    assert y.der == np.exp(2)


def test_sin():
    x = elementary_function(np.pi/2)
    y = elementary_function.sin(x)
    assert y.var == np.sin(np.pi/2)
    assert y.der == np.cos(np.pi/2)

    with pytest.raises(TypeError):
        y = elementary_function.sin("a")

    y = elementary_function.sin(np.pi/2)
    assert y.var == np.sin(np.pi/2)
    assert y.der == np.cos(np.pi/2)


def test_cos():
    x = elementary_function(np.pi/2)
    y = elementary_function.cos(x)
    assert y.var == np.cos(np.pi/2)
    assert y.der == -np.sin(np.pi/2)

    with pytest.raises(TypeError):
        y = elementary_function.cos("a")

    y = elementary_function.cos(np.pi)
    assert y.var == np.cos(np.pi)
    assert y.der == -np.sin(np.pi)


def test_tan():
    x = elementary_function(np.pi/3)
    y = elementary_function.tan(x)
    assert y.var == np.tan(np.pi/3)
    assert y.der == 1/np.cos(np.pi/3)**2

    with pytest.raises(TypeError):
        y = elementary_function.cos("a")

    y = elementary_function.tan(np.pi/3)
    assert y.var == np.tan(np.pi/3)
    assert y.der == 1/np.cos(np.pi/3)**2


def test_arcsin():
    x = elementary_function(1/2)
    y = elementary_function.arcsin(x)
    assert y.var == np.arcsin(1/2)
    assert y.der == 1 / np.sqrt(1 - (1/2) ** 2)

    with pytest.raises(TypeError):
        y = elementary_function.arcsin("a")

    y = elementary_function.arcsin(1/2)
    assert y.var == np.arcsin(1/2)
    assert y.der == 1 / np.sqrt(1 - (1/2) ** 2)


def test_arccos():
    x = elementary_function(1/2)
    y = elementary_function.arccos(x)
    assert y.var == np.arccos(1/2)
    assert y.der == -1 / np.sqrt(1 - (1/2) ** 2)

    with pytest.raises(TypeError):
        y = elementary_function.arccos("a")

    assert elementary_function.arccos(1/2) == np.arccos(1/2)

    # x = elementary_function(2)
    # with pytest.raises(ValueError):
    #     y = elementary_function.arccos(x)


def test_arctan():
    x = elementary_function(1/2)
    y = elementary_function.arctan(x)
    assert y.var == np.arctan(1/2)
    assert y.der == 1 / (1 + np.power(1/2, 2))

    with pytest.raises(TypeError):
        y = elementary_function.arctan("a")

    assert elementary_function.arctan(1/2) == np.arctan(1/2)


def test_sinh():
    x = elementary_function(1/2)
    y = elementary_function.sinh(x)
    assert y.var == np.sinh(1/2)
    assert y.der == np.cosh(1/2)

    with pytest.raises(TypeError):
        y = elementary_function.sinh("a")

    assert elementary_function.sinh(1/2) == np.sinh(1/2)


def test_cosh():
    x = elementary_function(1/2)
    y = elementary_function.cosh(x)
    assert y.var == np.cosh(1/2)
    assert y.der == np.sinh(1/2)

    with pytest.raises(TypeError):
        y = elementary_function.cosh("a")

    assert elementary_function.cosh(1/2) == np.cosh(1/2)


def test_tanh():
    x = elementary_function(1/2)
    y = elementary_function.tanh(x)
    assert y.var == np.tanh(1/2)
    assert y.der == 1 / np.cosh(1/2) ** 2

    with pytest.raises(TypeError):
        y = elementary_function.tanh("a")

    assert elementary_function.tanh(1/2) == np.tanh(1/2)


def test_logistic():
    x = elementary_function(2)
    y = elementary_function.logistic(x)
    assert y.var == 1 / (1 + np.exp(-2))
    assert y.der == 1 / (1 + np.exp(-2)) * (1 - 1 / (1 + np.exp(-2))) * x.der

    with pytest.raises(TypeError):
        y = elementary_function.logistic("a")
