from __future__ import print_function

import numpy as np

import time
import pickle
import config   # training configs
mlp_config = config.mlp_config

import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

from nets import MLP

np_output_file = 'exp_cifar10_mlp_{}'.format(mlp_config['context'])

if __name__ == '__main__':
    # get cifar10 data
    cifar10 = {
        'train_data': np.load('cifar10_training_data.npy').astype(np.float32) / 255.,
        'train_label': np.load('cifar10_training_label.npy').astype(np.int32),
        'test_data': np.load('cifar10_testing_data.npy').astype(np.float32) / 255.,
        'test_label': np.load('cifar10_testing_label.npy').astype(np.int32)
    }
    print('train_data', cifar10['train_data'].shape)
    print('train_label', cifar10['train_label'].shape)
    print('test_data', cifar10['test_data'].shape)
    print('test_label', cifar10['test_label'].shape)

    cifar10['train_data'] = np.reshape(cifar10['train_data'], (cifar10['train_data'].shape[0], -1))
    cifar10['test_data']  = np.reshape(cifar10['test_data'], (cifar10['test_data'].shape[0], -1))

    print(cifar10['train_data'].shape, cifar10['test_data'].shape, type(cifar10['train_label']), type(cifar10['test_label']))

    train = chainer.datasets.TupleDataset(cifar10['train_data'], cifar10['train_label'])
    test  = chainer.datasets.TupleDataset(cifar10['test_data'], cifar10['test_label'])

    print('cifar10 Data loaded')
    print('train_data, len:{}, shape:{}'.format(len(train), train[0][0].shape, train[0][1]))
    print('test_data, len:{}, shape:{}'.format(len(test), test[0][0].shape))
    print(type(train[0][0]), type(train[0][1]))

    # logging training time
    times = {}
    for l in mlp_config['layers']:
        for n in mlp_config['neurons']:
            for b in mlp_config['batch_size']:
                batch_size = b
                train_iter = iterators.SerialIterator(train, batch_size=batch_size, shuffle=True)
                test_iter = iterators.SerialIterator(test, batch_size=batch_size, repeat=False, shuffle=False)

                key = "layer{}-neuron{}-batch{}".format(l, n, b)
                times[key] = []

                for t in range(mlp_config['test_times']):
                    print("Current process: cifar10 {}, test {}".format(key, t))

                    model = L.Classifier(MLP(l, n, 10))
                    if mlp_config['context'] == 'gpu':
                        model.to_gpu()
        
                    optimizer = optimizers.Adam(alpha=0.0001)
                    optimizer.setup(model)

                    train_iter.reset()
                    test_iter.reset()
    
                    if mlp_config['context'] == 'gpu':
                        updater = training.StandardUpdater(train_iter, optimizer, device=0)
                    else:
                        updater = training.StandardUpdater(train_iter, optimizer)
                    trainer = training.Trainer(updater, (mlp_config['epochs'], 'epoch'), out='result')
                    
                    if mlp_config['context'] == 'gpu':
                        trainer.extend(extensions.Evaluator(test_iter, model, device=0))
                    else:
                        trainer.extend(extensions.Evaluator(test_iter, model))
                    trainer.extend(extensions.LogReport())
                    trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
                    # trainer.extend(extensions.ProgressBar())

                    ts = time.time()
                    trainer.run()
                    times[key].append(float(time.time()-ts))
    pickle.dump(times, open(np_output_file, 'wb'), True)
