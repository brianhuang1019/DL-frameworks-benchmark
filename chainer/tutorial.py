from __future__ import print_function

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

x_data = np.array([5], dtype=np.float32)
x = Variable(x_data)
y = x**2 - 2 * x + 1
print(y.data)
y.backward()
print(x.grad)

# chainer release intervalue's grad for memory efficiency
z = 7*x
y = x**2 - z + 1
y.backward()
assert z.grad == None
y.backward(retain_grad=True) # retain_grad=True to show grad of intervalue
print(z.grad)

# multi-element array grad
x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
y = x**2 - 2*x + 1
y.grad = np.ones((2, 3), dtype=np.float32)
y.backward()
print(x.grad)

f = L.Linear(3, 2)
print(f.W.data)
print(f.b.data)
f.cleargrads()

x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
y = f(x)
print(y.data)
