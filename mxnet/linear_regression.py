import mxnet as mx
from mxnet import nd
# from mxnet import autograd
import numpy as np

def net(X): 
    return nd.dot(X, w) + b

def square_loss(yhat, y): 
    return nd.mean((yhat - y) * (yhat-y))

def SGD(params, lr):    
    for param in params:
        param[:] = param - lr * param.grad

X = np.random.randn(10000,2)
y = 2* X[:,0] - 3.4 * X[:,1] + 4.2 + .01 * np.random.normal(size=10000)

batch_size = 4
train_data = mx.io.NDArrayIter(X, y, batch_size, shuffle=True)

batch = train_data.next()
print(batch.data[0])

w = nd.random_normal(shape=(2,1))
b = nd.random_normal(shape=1)

params = [w, b]

# for param in params:
#     param.attach_grad()

epochs = 2
ctx = mx.cpu()
moving_loss = 0.

for e in range(epochs):
    train_data.reset()
    for i, batch in enumerate(train_data):
        data = batch.data[0].as_in_context(ctx)
        label = batch.label[0].as_in_context(ctx).reshape((-1,1))
        # with autograd.record():
        output = net(data)
        loss = square_loss(output, label)
        loss.backward()
        SGD(params, .001)
        
        ##########################
        #  Keep a moving average of the losses
        ##########################
        if i == 0:
            moving_loss = np.mean(loss.asnumpy()[0])
        else:
            moving_loss = .99 * moving_loss + .01 * np.mean(loss.asnumpy()[0])
            
        if i % 500 == 0:
            print("Epoch %s, batch %s. Moving avg of loss: %s" % (e, i, moving_loss))
