from __future__ import print_function

import mxnet as mx
import numpy as np

import time
import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

np_output_file = 'mnist_mlp'

mnist = mx.test_utils.get_mnist()
print('MNIST Data loaded')
print('train_data', mnist['train_data'].shape)
print('train_label', mnist['train_label'].shape)
print('test_data', mnist['test_data'].shape)
print('test_label', mnist['test_label'].shape)

data = mx.sym.var('data')

# Flatten the data from 4-D shape into 2-D (batch_size, num_channel*width*height)
data = mx.sym.flatten(data=data)

# The first fully-connected layer and the corresponding activation function
fc1  = mx.sym.FullyConnected(data=data, num_hidden=128)
act1 = mx.sym.Activation(data=fc1, act_type="relu")

# The second fully-connected layer and the corresponding activation function
fc2  = mx.sym.FullyConnected(data=act1, num_hidden = 64)
act2 = mx.sym.Activation(data=fc2, act_type="relu")

# MNIST has 10 classes
fc3  = mx.sym.FullyConnected(data=act2, num_hidden=10)

# Softmax with cross entropy loss
mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')

# create a trainable module on CPU
mlp_model = mx.mod.Module(symbol=mlp, context=mx.cpu())

# training & logging time
times = []
for _ in range(10):
    batch_size = 100
    train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
    ts = time.time()
    mlp_model.fit(train_iter,  # train data
              eval_data=val_iter,  # validation data
              optimizer='sgd',  # use SGD to train
              optimizer_params={'learning_rate':0.1},  # use fixed learning rate
              eval_metric='acc',  # report accuracy during training
              batch_end_callback = mx.callback.Speedometer(batch_size, 100), # output progress for each 100 data batches
              num_epoch=10)  # train for at most 10 dataset passes
    times.append(float(time.time()-ts))
np.save(np_output_file, times)

# predict probability of mlp
test_iter = mx.io.NDArrayIter(mnist['test_data'], None, batch_size)
prob = mlp_model.predict(test_iter)
assert prob.shape == (10000, 10)

# predict accuracy of mlp
test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
acc = mx.metric.Accuracy()
mlp_model.score(test_iter, acc)
print(acc)
assert acc.get()[1] > 0.96
