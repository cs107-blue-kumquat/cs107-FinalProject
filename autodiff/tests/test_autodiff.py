import pytest
import numpy as np
from autodiff.autodiff import *


def test_init():
    x = Variable(2)
    assert x.var == 2
    assert x.der == 1

    x = Variable(2.)
    assert x.var == 2
    assert x.der == 1

    with pytest.raises(TypeError):
        x = Variable("abc")


def test_str():
    x = Variable(2)
    assert str(x) == "value = 2, derivative = 1"


def test_repr():
    x = Variable(2)
    assert repr(x) == "value = 2, derivative = 1"


def test_add():
    # add two Variable objects
    x = Variable(1)
    y = x + Variable(2)
    assert y.var == 3
    assert y.der == 2

    # add a int/float to Variable object
    x = Variable(1)
    y = 2. + x 
    assert y.var == 3
    assert y.der == 1

    # add a int/float to Variable object
    x = Variable(1)
    y = x + 2.
    assert y.var == 3
    assert y.der == 1
    
    # add a int/float to Variable object
    x = Variable(1)
    y = 2. + x + Variable(2)
    assert y.var == 5
    assert y.der == 2

    # add an invalid type of input to Variable object
    with pytest.raises(TypeError):
        x = Variable(1)
        y = x + "a"

    with pytest.raises(TypeError):
        x = Variable(1)
        y = "a" + x

def test_neg():
    x = Variable(1)
    y = -x
    assert y.var == -1
    assert y.der == -1


def test_sub():
    # subtract two Variable objects
    x = Variable(1)
    y = x - Variable(2)
    assert y.var == -1
    assert y.der == 0

    # subtract a int/float to Variable object
    x = Variable(1)
    y = x - 2.
    assert y.var == -1
    assert y.der == 1
    
    x = Variable(1)
    y = 2. - x
    assert y.var == 1
    assert y.der == -1

    # subtract an invalid type of input with Variable object
    with pytest.raises(TypeError):
        x = Variable(1)
        y = x - "a"


def test_mul():
    # multiply two Variable objects
    x = Variable(1)
    y = x * Variable(2)
    assert y.var == 2
    assert y.der == 3

    # multiply a int/float to Variable object
    x = Variable(1)
    y = x * 2.
    assert y.var == 2
    assert y.der == 2
    
    x = Variable(1)
    y = 2. * x
    assert y.var == 2
    assert y.der == 2

    x = Variable(1)
    y = x * -2.
    assert y.var == -2
    assert y.der == -2
    
    x = Variable(1)
    y = -2. * x
    assert y.var == -2
    assert y.der == -2

    # multiply an invalid type of input with Variable object
    with pytest.raises(TypeError):
        x = Variable(1)
        y = x * "a"


def test_truediv():
    # divide two Variable objects
    x = Variable(1)
    y = Variable(2) / x
    assert y.var == 2
    assert y.der == -1

    # divide a int/float by Variable object
    x = Variable(1)
    y = 5. / x
    assert y.var == 5
    assert y.der == 5
  
    # divide a int/float by Variable object
    x = Variable(1)
    y = x / 5.
    assert y.var == 1/5
    assert y.der == 1/5

    # divide Variable object by an invalid type of input 
    with pytest.raises(TypeError):
        x = Variable(1)
        y = x / "a"

    with pytest.raises(TypeError):
        x = Variable(1)
        y = "a" / x


def test_lt():
    x = Variable(1)
    y = Variable(2)
    assert (x < y) == True

    x = Variable(1)
    assert (x < 1) == False

    with pytest.raises(TypeError):
        x = Variable(1)
        check = "a" < x


def test_gt():
    x = Variable(1)
    y = Variable(2)
    assert (y > x) == True

    x = Variable(1)
    assert (x > 1) == False

    with pytest.raises(TypeError):
        x = Variable(1)
        check = "a" > x


def test_le():
    x = Variable(1)
    y = Variable(2)
    assert (x <= y) == True

    x = Variable(1)
    assert (x <= 1) == True

    with pytest.raises(TypeError):
        x = Variable(1)
        check = "a" <= x


def test_ge():
    x = Variable(1)
    y = Variable(2)
    assert (y >= x) == True

    x = Variable(1)
    assert (x >= 1) == True

    with pytest.raises(TypeError):
        x = Variable(1)
        check = "a" >= x


def test_eq():
    x = Variable(1)
    y = Variable(2)
    assert (y == x) == False
 
    with pytest.raises(TypeError):
        x = Variable(1)
        check = x == 1


def test_ne():
    x = Variable(1)
    y = Variable(2)
    assert (y != x) == True

    with pytest.raises(TypeError):
        x = Variable(1)
        check = x != 1


def test_abs():
    x = abs(Variable(1))
    assert x.var == 1
    assert x.der == 1

    y = abs(Variable(-2))
    assert y.var == 2
    assert y.der == 1


def test_pow():
    x = Variable(2)
    y = x ** 4
    assert y.var == 2 ** 4
    assert y.der == 4 * (2 ** 3)

    x = Variable(2)
    y = x ** -4
    assert y.var == 2 ** -4
    assert y.der == -4 * (2 ** -5)

    x = Variable(2)
    y = x ** x
    assert y.var == 2 ** 2
    assert y.der == 2 ** 2 * (np.log(2) * 1 / 1 + 2 / 2)

    x = Variable(2)
    y = x ** (x * 2)
    assert y.var == 2 ** (2 * 2)
    assert y.der == 2 ** (2 * 2) * (np.log(2) * 2 / 1 + (2 * 2) / 2)

    with pytest.raises(TypeError):
        x = Variable(1)
        check = x ** 1.2

    with pytest.raises(TypeError):
        x = Variable(1)
        check = x ** -1.2


def test_rpow():
    x = Variable(2)
    y = 4 ** x
    assert y.var == 4 ** 2
    assert y.der == 4 ** 2 * np.log(4)

    with pytest.raises(Exception):
        x = Variable(2)
        y = -4 ** x
        assert y.var == 4 ** 2

    x = Variable(2)
    y = 4 ** (x * 2)
    assert y.var == 4 ** (2 * 2)
    assert y.der == 4 ** (2 * 2) * np.log(4)

    with pytest.raises(ValueError):
        x = Variable(2)
        y = "a" ** x

def test_log():
    x = Variable(2)
    y = Variable.log(x)
    assert y.var == np.log(2)
    assert y.der == 1 / 2

    x = Variable(2)
    y = Variable.log(2 * x)
    assert y.var == 2 * np.log(2)
    assert y.der == 1 / 2

    with pytest.raises(TypeError):
        x = Variable(-2)
        y = Variable.log(x)

    with pytest.raises(TypeError):
        x = Variable(-2)
        y = Variable.log(2 * x)

    with pytest.raises(TypeError):
        y = Variable.log(2)


def test_sqrt():
    x = Variable(2)
    y = Variable.sqrt(x)
    assert y.var == np.sqrt(2)
    assert y.der == 1/2 * 2 ** (-1/2)

    x = Variable(2)
    y = Variable.sqrt(2 * x)
    assert y.var == np.sqrt(2 * 2)
    assert y.der == 1/ np.sqrt(2) * 1/2 * 2 ** (-1/2)

    with pytest.raises(ValueError):
        x = Variable(-2)
        y = Variable.sqrt(x)

    with pytest.raises(TypeError):
        y = Variable.sqrt("a")

    with pytest.raises(TypeError):
        y = Variable.sqrt(2)


def test_exp():
    x = Variable(2)
    y = Variable.exp(x)
    assert y.var == np.exp(2)
    assert y.der == np.exp(2)

    x = Variable(2)
    y = Variable.exp(2 * x)
    assert y.var == np.exp(2 * 2)
    assert y.der == 2 * np.exp(2 * 2)

    with pytest.raises(TypeError):
        y = Variable.exp("a")

    y = Variable.exp(2)
    assert y.var == np.exp(2)
    assert y.der == np.exp(2)


def test_sin():
    x = Variable(np.pi/2)
    y = Variable.sin(x)
    assert y.var == np.sin(np.pi/2)
    assert y.der == np.cos(np.pi/2)

    x = Variable(np.pi/2)
    y = Variable.sin(2 * x)
    assert y.var == np.sin(2 * np.pi/2)
    assert y.der == 2 * np.cos(2 * np.pi/2)

    with pytest.raises(TypeError):
        y = Variable.sin("a")

    y = Variable.sin(np.pi/2)
    assert y.var == np.sin(np.pi/2)
    assert y.der == np.cos(np.pi/2)


def test_cos():
    x = Variable(np.pi/2)
    y = Variable.cos(x)
    assert y.var == np.cos(np.pi/2)
    assert y.der == -np.sin(np.pi/2)

    with pytest.raises(TypeError):
        y = Variable.cos("a")

    y = Variable.cos(np.pi)
    assert y.var == np.cos(np.pi)
    assert y.der == -np.sin(np.pi)


def test_tan():
    x = Variable(np.pi/3)
    y = Variable.tan(x)
    assert y.var == np.tan(np.pi/3)
    assert y.der == 1/np.cos(np.pi/3)**2

    with pytest.raises(TypeError):
        y = Variable.tan("a")

    y = Variable.tan(np.pi/3)
    assert y.var == np.tan(np.pi/3)
    assert y.der == 1/np.cos(np.pi/3)**2


def test_arcsin():
    x = Variable(1/2)
    y = Variable.arcsin(x)
    assert y.var == np.arcsin(1/2)
    assert y.der == 1 / np.sqrt(1 - (1/2) ** 2)

    with pytest.raises(TypeError):
        y = Variable.arcsin("a")

    with pytest.raises(TypeError):
        x = Variable(-2)
        y = Variable.arcsin(x)

    y = Variable.arcsin(1/2)
    assert y.var == np.arcsin(1/2)
    assert y.der == 1 / np.sqrt(1 - (1/2) ** 2)


def test_arccos():
    x = Variable(1/2)
    y = Variable.arccos(x)
    assert y.var == np.arccos(1/2)
    assert y.der == -1 / np.sqrt(1 - (1/2) ** 2)

    with pytest.raises(TypeError):
        y = Variable.arccos("a")

    with pytest.raises(TypeError):
        x = Variable(-2)
        y = Variable.arccos(x)

    assert Variable.arccos(1/2) == np.arccos(1/2)

    # x = Variable(2)
    # with pytest.raises(ValueError):
    #     y = Variable.arccos(x)


def test_arctan():
    x = Variable(1/2)
    y = Variable.arctan(x)
    assert y.var == np.arctan(1/2)
    assert y.der == 1 / (1 + np.power(1/2, 2))

    with pytest.raises(TypeError):
        y = Variable.arctan("a")

    assert Variable.arctan(1/2) == np.arctan(1/2)


def test_sinh():
    x = Variable(1/2)
    y = Variable.sinh(x)
    assert y.var == np.sinh(1/2)
    assert y.der == np.cosh(1/2)

    with pytest.raises(TypeError):
        y = Variable.sinh("a")

    assert Variable.sinh(1/2) == np.sinh(1/2)


def test_cosh():
    x = Variable(1/2)
    y = Variable.cosh(x)
    assert y.var == np.cosh(1/2)
    assert y.der == np.sinh(1/2)

    with pytest.raises(TypeError):
        y = Variable.cosh("a")

    assert Variable.cosh(1/2) == np.cosh(1/2)


def test_tanh():
    x = Variable(1/2)
    y = Variable.tanh(x)
    assert y.var == np.tanh(1/2)
    assert y.der == 1 / np.cosh(1/2) ** 2

    with pytest.raises(TypeError):
        y = Variable.tanh("a")

    assert Variable.tanh(1/2) == np.tanh(1/2)


def test_sigmoid():
    x = Variable(2)
    y = Variable.sigmoid(x)
    assert y.var == 1 / (1 + np.exp(-2))
    assert y.der == 1 / (1 + np.exp(-2)) * (1 - 1 / (1 + np.exp(-2))) * x.der

    with pytest.raises(TypeError):
        y = Variable.sigmoid("a")
