from __future__ import print_function

import time
import pickle
import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

import config   # training configs
cnn_config = config.cnn_config

import mxnet as mx
import numpy as np

np_output_file = 'exp_cifar10_cnn_{}'.format(cnn_config['context'])

def net(n_config):
    print('net config', n_config)

    act_type = n_config['act']
    kernal = (n_config['kernal_v'], n_config['kernal_h'])
    depths = n_config['depths']

    pool_type = n_config['pool']['type']
    pool_kernal = (n_config['pool']['kernal_v'], n_config['pool']['kernal_h'])
    pool_stride = (n_config['pool']['stride_v'], n_config['pool']['stride_h'])

    fc_neurons = n_config['fc_neurons']

    data = mx.sym.Variable('data')

    # first conv layer
    conv = mx.sym.Convolution(data=data, kernel=kernal, num_filter=depths[0])
    act = mx.sym.Activation(data=conv, act_type=act_type)
    pool = mx.sym.Pooling(data=act, pool_type=pool_type, kernel=pool_kernal, stride=pool_stride)
    
    # mid conv layer(s)
    for filters in depths[1:]:
        conv = mx.sym.Convolution(data=pool, kernel=kernal, num_filter=filters)
        act = mx.sym.Activation(data=conv, act_type=act_type)
        pool = mx.sym.Pooling(data=act, pool_type=pool_type, kernel=pool_kernal, stride=pool_stride)
    
    # first fc layer
    flatten = mx.sym.Flatten(data=pool)
    fc = mx.sym.FullyConnected(data=flatten, num_hidden=fc_neurons[0])
    act = mx.sym.Activation(data=fc, act_type=act_type)

    # mid fc layer(s)
    for n in fc_neurons[1:-1]:
        fc = mx.sym.FullyConnected(data=act, num_hidden=n)
        act = mx.sym.Activation(data=fc, act_type=act_type)
    
    # last fc layer
    fc = mx.sym.FullyConnected(data=act, num_hidden=fc_neurons[-1])
    
    # softmax loss
    cnn = mx.sym.SoftmaxOutput(data=fc, name='softmax')

    # create a trainable module on CPU/GPU
    if cnn_config['context'] == 'cpu':
        model = mx.mod.Module(symbol=cnn, context=mx.cpu())
    elif cnn_config['context'] == 'gpu':
        model = mx.mod.Module(symbol=cnn, context=mx.gpu())
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
    for _, net_config in enumerate(cnn_config['nets']):
        for b in cnn_config['batch_size']:
            batch_size = b
            train_iter = mx.io.NDArrayIter(cifar10['train_data'], cifar10['train_label'], batch_size, shuffle=True)
            val_iter = mx.io.NDArrayIter(cifar10['test_data'], cifar10['test_label'], batch_size)

            key = "{}-batch{}".format(net_config['alias'], b)
            times[key] = []

            for t in range(cnn_config['test_times']):
                print("Cifar10 Current process: {}, test {}".format(key, t))
                train_iter.reset()
                val_iter.reset()
                cnn_model = net(net_config)
                ts = time.time()
                cnn_model.fit(train_iter,  # train data
                    eval_data=val_iter,  # validation data
                    optimizer=cnn_config['optimizer'],  # use SGD to train
                    optimizer_params={'learning_rate': 0.0001},  # use fixed learning rate
                    eval_metric='acc',  # report accuracy during training
                    batch_end_callback = mx.callback.Speedometer(batch_size, 100), # output progress for each 100 data batches
                    num_epoch=cnn_config['epochs'])  # train for at most 10 dataset passes
                times[key].append(float(time.time()-ts))

    pickle.dump(times, open(np_output_file, 'wb'), True)
    
    # predict probability of mlp
    test_iter = mx.io.NDArrayIter(cifar10['test_data'], None, batch_size)
    prob = cnn_model.predict(test_iter)
    assert prob.shape == (10000, 10)
    
    # predict accuracy of mlp
    test_iter = mx.io.NDArrayIter(cifar10['test_data'], cifar10['test_label'], batch_size)
    acc = mx.metric.Accuracy()
    cnn_model.score(test_iter, acc)
    print(acc)
