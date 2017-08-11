from __future__ import print_function

import time
import pickle
import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

import util
import sys
sys.path.append('..')
import config   # training configs
mlp_config = config.mlp_config

import mxnet as mx
import numpy as np

exp_output_file = 'exp_cifar10_mlp_{}'.format(mlp_config['context'])
if mlp_config['context'] == 'multi-gpu':
    exp_output_file += "-{}".format(mlp_config['gpus'])

def net(layers, neurons):
    data = mx.sym.Variable('data')

    # Flatten the data from 4-D shape into 2-D (batch_size, num_channel*width*height)
    data = mx.sym.Flatten(data=data)

    # The first fully-connected layer and the corresponding activation function
    fc  = mx.sym.FullyConnected(data=data, num_hidden=neurons)
    act = mx.sym.Activation(data=fc, act_type="relu")

    for _ in range(layers-2):
        fc  = mx.sym.FullyConnected(data=act, num_hidden=neurons)
        act = mx.sym.Activation(data=fc, act_type="relu")

    # Cifar10 has 10 classes
    fc  = mx.sym.FullyConnected(data=act, num_hidden=10)

    # Softmax with cross entropy loss
    mlp = mx.sym.SoftmaxOutput(data=fc, name='softmax')

    # create a trainable module on CPU/GPU
    if mlp_config['context'] == 'cpu':
        model = mx.mod.Module(symbol=mlp, context=mx.cpu())
    elif mlp_config['context'] == 'gpu':
        model = mx.mod.Module(symbol=mlp, context=mx.gpu())
    elif mlp_config['context'] == 'multi-gpu':
        assert mlp_config['gpus'] > 1
        gpus = [mx.gpu(i) for i in range(mlp_config['gpus'])]
        model = mx.mod.Module(symbol=mlp, context=gpus)
    else:
        pass

    return model

if __name__ == '__main__':
    # get cifar10 data
    cifar10 = {
        'train_data': np.load('../data/cifar10_training_data.npy'),
        'train_label': np.load('../data/cifar10_training_label.npy'),
        'test_data': np.load('../data/cifar10_testing_data.npy'),
        'test_label': np.load('../data/cifar10_testing_label.npy')
    }
    print('cifar10 Data loaded')
    print('train_data', cifar10['train_data'].shape)
    print('train_label', cifar10['train_label'].shape)
    print('test_data', cifar10['test_data'].shape)
    print('test_label', cifar10['test_label'].shape)

    # logging training time
    times = {}
    for l in mlp_config['layers']:
        for n in mlp_config['neurons']:
            for b in mlp_config['batch_size']:
                batch_size = b
                train_iter = mx.io.NDArrayIter(cifar10['train_data'], cifar10['train_label'], batch_size, shuffle=True)
                val_iter = mx.io.NDArrayIter(cifar10['test_data'], cifar10['test_label'], batch_size)

                key = "layer{}-neuron{}-batch{}".format(l, n, b)
                times[key] = []

                for t in range(mlp_config['test_times']):
                    print("Cifar10 Current process: {}, test {}".format(key, t))
                    train_iter.reset()
                    val_iter.reset()
                    mlp_model = net(l, n)
                    ts = time.time()
                    mlp_model.fit(train_iter,  # train data
                        eval_data=val_iter,  # validation data
                        optimizer=mlp_config['optimizer'],  # use SGD to train
                        optimizer_params={'learning_rate': 0.0001},  # use fixed learning rate
                        eval_metric='acc',  # report accuracy during training
                        batch_end_callback = mx.callback.Speedometer(batch_size, util.decideInterval(cifar10['train_data'].shape[0], batch_size)), # output progress for each 100 data batches
                        num_epoch=mlp_config['epochs'])  # train for at most 10 dataset passes
                    times[key].append(float(time.time()-ts))

    pickle.dump(times, open(exp_output_file, 'wb'), True)
    
    # predict probability of mlp
    test_iter = mx.io.NDArrayIter(cifar10['test_data'], None, batch_size)
    prob = mlp_model.predict(test_iter)
    assert prob.shape == (10000, 10)
    
    # predict accuracy of mlp
    test_iter = mx.io.NDArrayIter(cifar10['test_data'], cifar10['test_label'], batch_size)
    acc = mx.metric.Accuracy()
    mlp_model.score(test_iter, acc)
    print(acc)
