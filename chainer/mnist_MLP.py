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

np_output_file = 'exp_mnist_mlp_{}'.format(mlp_config['context'])

if __name__ == '__main__':
    train, test = datasets.get_mnist()
    print('mnist Data loaded')
    print('train_data, len:{}, shape:{}'.format(len(train), train[0][0].shape))
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
                    print("Current process: MLP {}, test {}".format(key, t))

                    model = L.Classifier(MLP(l, n, 10))  # the input size, 784, is inferred
                    if mlp_config['context'] == 'gpu':
                        model.to_gpu()
        
                    optimizer = optimizers.Adam()
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
