import pytest
import numpy as np
from BlueKumquatAutoDiff.autodiff import *


def test_node_init():
    x = Node(2)
    assert x.var == 2
    assert x.partials() == 1

    x = Node(2.)
    assert x.var == 2
    assert x.partials() == 1

    with pytest.raises(TypeError):
        x = Node("abc")


def test_node_get_derivatives():
    x = Node(2)
    z = Node(3)
    y = x * z
    assert y.get_derivatives([x,z])[0] == 6
    assert y.get_derivatives([x,z])[1][0] == 3
    assert y.get_derivatives([x,z])[1][1] == 2
    

def test_node_partials():
    pass


def test_node_str():
    x = Node(2)
    assert str(x) == "value = 2, derivative = 1"


def test_node_repr():
    x = Node(2)
    assert repr(x) == "value = 2, derivative = 1"


def test_node_add():
    # add two Node objects
    x = Node(1)
    y = x+Node(2)
    assert y.var == 3
    assert y.partials() == 1

    # add a int/float to Node object
    x = Node(1)
    y = 2. + x 
    assert y.var == 3
    assert y.partials() == 1.

    # add a int/float to Node object
    x = Node(1)
    y = x + 2.
    assert y.var == 3
    assert y.partials() == 1.
    
    # add a int/float to Node object
    x = Node(1)
    y = 2. + x + Node(2)
    assert y.var == 5
    assert y.partials() == 1

    # add an invalid type of input to Node object
    with pytest.raises(TypeError):
        x = Node(1)
        y = x + "a"

    with pytest.raises(TypeError):
        x = Node(1)
        y = "a" + x


def test_node_neg():
    x = Node(1)
    y = -x
    assert y.var == -1
    assert y.partials() == 1
    assert x.partials() == -1


def test_node_sub():
    # subtract two Node objects
    x = Node(1)
    y = x - Node(2)
    assert y.var == -1
    assert y.partials() == 1

    # subtract a int/float to Node object
    x = Node(1)
    y = x - 2.
    assert y.var == -1
    assert y.partials() == 1
    
    x = Node(1)
    y = 2. - x
    assert y.var == 1
    assert y.partials() == 1
    assert x.partials() == -1

    # subtract an invalid type of input with Node object
    with pytest.raises(TypeError):
        x = Node(1)
        y = x - "a"


def test_node_mul():
    # multiply two Node objects
    x = Node(1)
    y = x * Node(2)
    assert y.var == 2
    assert y.partials() == 1
    assert x.partials() == 2

    # multiply a int/float to Node object
    x = Node(1)
    y = x * 2.
    assert y.var == 2
    assert y.partials() == 1
    assert x.partials() == 2
    
    x = Node(1)
    y = 2. * x
    assert y.var == 2
    assert x.partials() == 2
    assert y.partials() == 1

    x = Node(1)
    y = x * -2.
    assert y.var == -2
    assert x.partials() == -2
    assert y.partials() == 1
    
    x = Node(1)
    y = -2. * x
    assert y.var == -2
    assert x.partials() == -2
    assert y.partials() == 1

    # multiply an invalid type of input with Node object
    with pytest.raises(TypeError):
        x = Node(1)
        y = x * "a"


def test_node_truediv():
    # divide two Node objects
    x = Node(1)
    y = Node(2) / x
    assert y.var == 2
    assert y.partials() == 1
    assert x.partials() == -2

    # divide a int/float by Node object
    x = Node(1)
    y = 5. / x
    assert y.var == 5
    assert x.partials() == -5
  
    # divide a int/float by Node object
    x = Node(1)
    y = x / 5.
    assert y.var == 1/5
    assert x.partials() == 1/5
    assert y.partials() == 1

    # divide Node object by an invalid type of input 
    with pytest.raises(TypeError):
        x = Node(1)
        y = x / "a"

    with pytest.raises(TypeError):
        x = Node(1)
        y = "a" / x


def test_node_lt():
    x = Node(1)
    y = Node(2)
    assert (x < y) == True

    x = Node(1)
    assert (x < 1) == False

    with pytest.raises(TypeError):
        x = Node(1)
        check = "a" < x


def test_node_gt():
    x = Node(1)
    y = Node(2)
    assert (y > x) == True

    x = Node(1)
    assert (x > 1) == False

    with pytest.raises(TypeError):
        x = Node(1)
        check = "a" > x


def test_node_le():
    x = Node(1)
    y = Node(2)
    assert (x <= y) == True

    x = Node(1)
    assert (x <= 1) == True

    with pytest.raises(TypeError):
        x = Node(1)
        check = "a" <= x


def test_node_ge():
    x = Node(1)
    y = Node(2)
    assert (y >= x) == True

    x = Node(1)
    assert (x >= 1) == True

    with pytest.raises(TypeError):
        x = Node(1)
        check = "a" >= x


def test_node_eq():
    x = Node(1)
    y = Node(2)
    assert (y == x) == False
 
    with pytest.raises(TypeError):
        x = Node(1)
        check = x == 1


def test_node_ne():
    x = Node(1)
    y = Node(2)
    assert (y != x) == True

    with pytest.raises(TypeError):
        x = Node(1)
        check = x != 1


def test_node_abs():
    x = abs(Node(1))
    assert x.var == 1
    assert x.partials() == 1

    y = abs(Node(-2))
    assert y.var == 2
    assert y.partials() == 1


def test_node_pow():
    x = Node(2)
    y = x ** 4
    assert y.var == 2 ** 4
    assert x.partials() == 4 * (2 ** 3)
    assert y.partials() == 1

    x = Node(2)
    y = x ** -4
    assert y.var == 2 ** -4
    assert x.partials() == -4 * (2 ** -5)
    assert y.partials() == 1

    x = Node(2)
    y = x ** x
    assert y.var == 2 ** 2
    assert x.partials() == 2 ** 2 * (np.log(2) * 1 / 1 + 2 / 2)
    assert y.partials() == 1

    x = Node(2)
    y = x ** (x * 2)
    assert y.var == 2 ** (2 * 2)
    assert x.partials() == 2 ** (2 * 2) * (np.log(2) * 2 / 1 + (2 * 2) / 2)
    assert y.partials() == 1

    with pytest.raises(TypeError):
        x = Node(1)
        check = x ** 1.2

    with pytest.raises(TypeError):
        x = Node(1)
        check = x ** -1.2


def test_node_rpow():
    x = Node(2)
    y = 4 ** x
    assert y.var == 4 ** 2
    assert x.partials() == 4 ** 2 * np.log(4)
    assert y.partials() == 1

    with pytest.raises(Exception):
        x = Node(2)
        y = -4 ** x
        assert y.var == 4 ** 2

    x = Node(2)
    y = 4 ** (x * 2)
    assert y.var == 4 ** (2 * 2)
    assert x.partials() == 2 ** (4 * 2 + 1) * np.log(4)

    with pytest.raises(ValueError):
        x = Node(2)
        y = "a" ** x

def test_node_log():
    x = Node(2)
    y = Node.log(x)
    assert y.var == np.log(2)
    assert x.partials() == 1 / 2
    assert y.partials() == 1

    x = Node(2)
    y = Node.log(2 * x)
    assert y.var == 2 * np.log(2)
    assert x.partials() == 1 / 2
    assert y.partials() == 1

    with pytest.raises(TypeError):
        x = Node(-2)
        y = Node.log(x)

    with pytest.raises(TypeError):
        x = Node(-2)
        y = Node.log(2 * x)

    with pytest.raises(TypeError):
        y = Node.log(2)


def test_node_sqrt():
    x = Node(2)
    y = Node.sqrt(x)
    assert y.var == np.sqrt(2)
    assert x.partials() == 1/2 * 2 ** (-1/2)

    x = Node(2)
    y = Node.sqrt(2 * x)
    assert y.var == np.sqrt(2 * 2)
    assert x.partials() == 1/ np.sqrt(2) * 2 ** (-1/2)
    with pytest.raises(ValueError):
        x = Node(-2)
        y = Node.sqrt(x)

    with pytest.raises(TypeError):
        y = Node.sqrt("a")

    with pytest.raises(TypeError):
        y = Node.sqrt(2)


def test_node_exp():
    x = Node(2)
    y = Node.exp(x)
    assert y.var == np.exp(2)
    assert x.partials() == np.exp(2)

    x = Node(2)
    y = Node.exp(2 * x)
    assert y.var == np.exp(2 * 2)
    assert x.partials() == 2 * np.exp(2 * 2)

    with pytest.raises(TypeError):
        y = Node.exp("a")

    y = Node.exp(2)
    assert y == np.exp(2)


def test_node_sin():
    x = Node(np.pi/2)
    y = Node.sin(x)
    assert y.var == np.sin(np.pi/2)
    assert x.partials() == np.cos(np.pi/2)

    x = Node(np.pi/2)
    y = Node.sin(2 * x)
    assert y.var == np.sin(2 * np.pi/2)
    assert x.partials() == 2 * np.cos(2 * np.pi/2)

    with pytest.raises(TypeError):
        y = Node.sin("a")

    y = Node.sin(np.pi/2)
    assert y == np.sin(np.pi/2)


def test_node_cos():
    x = Node(np.pi/2)
    y = Node.cos(x)
    assert y.var == np.cos(np.pi/2)
    assert x.partials() == -np.sin(np.pi/2)

    with pytest.raises(TypeError):
        y = Node.cos("a")

    y = Node.cos(np.pi)
    assert y == np.cos(np.pi)


def test_node_tan():
    x = Node(np.pi/3)
    y = Node.tan(x)
    assert y.var == np.tan(np.pi/3)
    assert x.partials() == 1/np.cos(np.pi/3)**2

    with pytest.raises(TypeError):
        y = Node.tan("a")

    y = Node.tan(np.pi/3)
    assert y == np.tan(np.pi/3)


def test_node_arcsin():
    x = Node(1/2)
    y = Node.arcsin(x)
    assert y.var == np.arcsin(1/2)
    assert x.partials() == 1 / np.sqrt(1 - (1/2) ** 2)

    with pytest.raises(TypeError):
        y = Node.arcsin("a")

    with pytest.raises(TypeError):
        x = Node(-2)
        y = Node.arcsin(x)

    y = Node.arcsin(1/2)
    assert y == np.arcsin(1/2)


def test_node_arccos():
    x = Node(1/2)
    y = Node.arccos(x)
    assert y.var == np.arccos(1/2)
    assert x.partials() == -1 / np.sqrt(1 - (1/2) ** 2)

    with pytest.raises(TypeError):
        y = Node.arccos("a")

    with pytest.raises(TypeError):
        x = Node(-2)
        y = Node.arccos(x)

    assert Node.arccos(1/2) == np.arccos(1/2)


def test_node_arctan():
    x = Node(1/2)
    y = Node.arctan(x)
    assert y.var == np.arctan(1/2)
    assert x.partials() == 1 / (1 + np.power(1/2, 2))

    with pytest.raises(TypeError):
        y = Node.arctan("a")

    assert Node.arctan(1/2) == np.arctan(1/2)


def test_node_sinh():
    x = Node(1/2)
    y = Node.sinh(x)
    assert y.var == np.sinh(1/2)
    assert x.partials() == np.cosh(1/2)

    with pytest.raises(TypeError):
        y = Node.sinh("a")

    assert Node.sinh(1/2) == np.sinh(1/2)


def test_node_cosh():
    x = Node(1/2)
    y = Node.cosh(x)
    assert y.var == np.cosh(1/2)
    assert x.partials() == np.sinh(1/2)

    with pytest.raises(TypeError):
        y = Node.cosh("a")

    assert Node.cosh(1/2) == np.cosh(1/2)


def test_node_tanh():
    x = Node(1/2)
    y = Node.tanh(x)
    assert y.var == np.tanh(1/2)
    assert y.partials() == 1
    with pytest.raises(TypeError):
        y = Node.tanh("a")

    assert Node.tanh(1/2) == np.tanh(1/2)


def test_node_sigmoid():
    x = Node(2)
    y = Node.sigmoid(x)
    assert y.var == 1 / (1 + np.exp(-2))
    assert x.partials() == 1 / (1 + np.exp(-2)) * (1 - 1 / (1 + np.exp(-2)))

    with pytest.raises(TypeError):
        y = Node.sigmoid("a")
