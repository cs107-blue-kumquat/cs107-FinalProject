# Milestone 1 (as of 10/21/2021)

## Introduction

The software that we are going to build aims to achieve automatic differentiation. Finding the derivative of a function helps people understand the rate of change of a variable. In some real-life scenarios, we could compute the velocity of an object and make models to simulate and optimize the object movement. Automatic Differentiation (AD) could help speed up the derivative computation by applying chain rule iteratively and performing arithmetic operations. 

AD is derived from symbolic differentiation and numerical differentiation to mitigate the potential errors brought by those algorithms. Symbolic differentiation gives exact computation with variables taken into the expressions, which would take a lot of time to evaluate and there are also issues related to the order of operations and the feasibility of code implementation. Numerical differentiation estimates the derivative of a function using Newton's method to approximate the slope of a nearby secant line, but the rounding error also arises. Both of the differentiation methods need to derive the derivative of the original function as an intermediate step, but AD could directly decompose this process as a list of elementary functions and evaluate them with the same accuracy.

## Background

### Computational graph

Computational graph is a convenient way to represent mathematical expressions and visualize the process of evaluation. Each of the variables would be regarded as nodes in a graph, and the types of operations as edges connecting the nodes. The output of a pair of nodes would generate a new node, which would be used in the next steps of evaluation. It starts with the original variables and their derivatives as input nodes, and the goal is to build a graph and compute the target expression.
$$
\mbox{Example: } f(x) = (\sin x)^2
$$


![Drawing](https://tva1.sinaimg.cn/large/008i3skNgy1gvmvhlerqej61h90u0t9l02.jpg)

### Chain rule

Chain rule is very useful in computing the derivative when the original expression is complicated. It decomposes the complex function into simple functions and then takes the derivative of a composite of them, the product is equivalent to the derivative of the original function. If $g$ is a function differentiable at $a$ and f is a function differentiable at $f(a)$, so the composite function $f · g$ is differentiable at $a$. 
$$
\frac{dy}{da} =\frac{dy}{dg}\frac{dg}{da}
$$

### Forward differentiation

In the chain rule introduced above, the forward differentiation was presented, where traversal starts from the original independent variables, computes the derivative, gets the expression of the inner function, and computes the derivative recursively. When there is more than one variable in the forward differentiation step, their derivatives need to be taken with respect to each variable once so that the gradient could be calculated correctly.

## How to use AutomaticDifferentiation Package

The user’s interactions with our package will be for educational and mathematical purposes as the use case of Automatic Differentiation is very flexible. Students can use our package to learn more about Automatic Differentiation and computational graphs. At-home users can use this package to solve basic differential equations like a calculator and companies can implement this package as a building block to conquer more complex differential equations. These are a few of the ways that users will interact with the AutomaticDifferentiation package. 

Users should import this package through Python Package Index PyPi, as this is how we are planning to install and distribute packages. This is a remote server from which all users can fetch different packages. The user can use a python tool called “pip” and can call the command:

```python
pip install automatic-differentiation
```

in order to import our package. For some elementary functions we are not going to use an existing package and we created the code of the derivatives for some elementary functions like trigonometric functions.

in order to import our package. For some elementary functions we are not going to use an existing package and we created the code of the derivatives for some elementary functions like trigonometric functions.

Users can instantiate AD objects by: 

```python
AD_obj  = AutomaticDifferentiation(name, elementary_function, numerical_value)
```

`name` is a string that is an identifier about what function we are computing. `elementary_function` is a basic function like sin, cos, log etc. `numerical_value` is a number .

## Software Organization

This software package is organized with the following hierarchy. First, files for understanding the uses and limitations of the package will be listed in the main project directory. This will include the LICENSE file, README file, and source code directory, containing the actual implementation of the software. In the source code file, we will have a main initialization of object parameters in the `__init.py__` file along with other general use files. Next, we will be implementing sub packages like a package for forward mode differentiation, reverse model differentiation, another subpackage for the “extension” sub-package of our choosing, like higher-order differentiation or mixed-mode differentiation, and perhaps another sub-package for elementary functions, depending on the need. Within each of these sub-packages will likely contain other files for parameter initialization and specific methods for calculating each subpackage function.

Our test suite will exist on both CodeCov for looking at code coverage and TravisCI for use in automatic testing. We will distribute our package through PyPI which is a server for python software packages, allowing for easy uploading and distribution of our package. The build system dependencies will be stored in a .toml file and contained within will be build requirements for our project and setup tools will be located in a .cfg file. The framework we will work off of will likely be PEP518 as this allows easy access and installation through pip with well-defined setup instructions. Moreover, we are actively thinking about other considerations that may benefit our project. 

## Implementation

In this project we will create a package for automatic differentiation. Automatic differentiation is not the same as symbolic differentiation nor numerical differentiation. The forward mode of automatic differentiation applies the chain rule to each basic operation and the gradient is obtained by multiplying the individual pieces together. 

### What are the core data structures?

The core data structures are dependent on how we choose to implement the forward mode of automatic differentiation. If implemented using a tuple method where each $v_i$ and $D_p v_i$ are stored but updated since we know that once a child node is evaluated then its parent node(s) are no longer needed so long as there are no more child nodes to be evaluated for those parent nodes. This would be best used in functions of f with a sole output. 

If implemented using dual numbers we would want to create a class in the same manner we reviewed creating a class for complex variables in class that would store the real part of the value $v_i$ as the primal trace which is best visualized in a computational graph and the dual part is the part corresponding to the tangent trace. This method of implementation would allow us to store values for $v_i$s and it will allow us to access the real part and the dual part and allow us to adjust how operations would best be implemented to service our user.

If implemented using the Jacobian, we would want to store partials as a matrix or vector and p as a seed vector, then we could access different values given what we set our seed vector p to be. Our forward mode automatic differentiation would compute the dot product of our Jacobian matrix and seed vector $\mathbf{p}$. 

### What classes will you implement? What method and name attributes will your classes have?

We would want a class that initializes our forward mode of automatic differentiation, by isolating the function, the variables, the values to evaluate at. Init would have name, elementary function, and the numerical value it should be isolated at. 

We will need classes that compute derivatives and perform symbolic differentiation in order to compute the partial derivatives needed later. We need a class for partial derivatives with respect to each variable. This class would be dependent on the given function and variables, so in an example of this you would pass in variables and the function from what is given

```python
class Partials(self):
# To find partials I was thinking self.function and self.variable and perhaps in a for loop for each function in self.function and within that loop for each variable in self.variable then store as the dual part in dual number or store in corresponding location in Jacobian matrix or save in some variable to store in tuple depending on the implementation we choose.
	p1 # returns partial with respect to variable 1
	p2 # returns partial with respect to variable 2
	…
	pn # returns partial with respect to variable n
	return p1, p2, …, pn
```

### What external dependencies will you rely on?

We would have numpy for vectors, matrices, and elementary functions. 

### How will you deal with elementary functions like sin, sqrt, log, exp, and all others?

It makes sense to code derivatives of known functions in if we are not allowed to use an existing package and then use the library we generate for automatic differentiation. 

- derivative of $e^x$ is $e^x$
- Derivative of $x^n$ is $nx^{n-1}$

- Derivatives of trigonometric functions
- Derivatives of logarithmic functions
- Derivatives of root functions by rewriting roots as power functions
- Product rule, quotient rule, chain rule, etc. 

Jacobian, Newton root code

```python
def f(x,y):
	fx = # function for fx  
	fy = # function for fy  return [fx, fy]


def J(x,y):
	dxx = # derivative of fx with respect to x
  dxy = # derivative of fx with respect to y
  dyx = # derivative of fy with respect to x
  dyy = # derivative of fy with respect to y
  return [dxx, dxy, dyx, dyy]


def Newton_root(x,y):
  n = 0
  while n < 10000:
    dxx, dxy, dyx, dyy = J(x,y)
    fx, fy = f(x,y)
    det = 1 / ((dxx * dyy) - (dxy * dyx))
    x = x -(dyy * fx - dxy * fy)*det
    y = y - (-dyx* fx + dxx * fy)*det
    n += 1
    return x,y
```

## License

Automatic differentiation is a topic in CS that has already been implemented and is not unique to this particular package. Thus, we will not have to deal with patents. We want all people who are interested in using Automated differentiation to be able to see our code and to use the data structures and methods we have implemented. It can then be utilized for their own personal use like computing the automatic differentiation in their work or school assignments. We also want to ensure that the maximum number of users are able to use this package and for the license to have a flexible use case for a variety of scenarios. We have chosen a CopyLeft software license for this package. A CopyLeft license is suitable for this project because CopyLeft ensures that a program, as well as all of its modifications and extended versions, are free as well. Our aim is to encourage the extension of the Automatic Differentiation work done in our package to increase the accessibility of our work. We have chosen the MIT license as it is much more permissive and allows companies to make changes to the package and not be required to share these changes with the public. The MIT license is suitable because it caters to the user’s personal use case. There are many ways that our package can be utilized and we need a license that is flexible enough to accommodate all use cases. 

Lastly, we can increase the publicity of our package by one day building a separate website that contains the use cases, examples, and documentation for our package. We encourage our users to cite our package in academic papers if used by writing a similar excerpt in your works cited in order to increase the publicity of this package: 

>  The authors thank the developers of AutomaticDifferentiation (BlueKumquat et al., 2021) for making their > code available on a free and open-source basis.

