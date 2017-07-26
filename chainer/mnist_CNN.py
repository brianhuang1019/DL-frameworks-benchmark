from __future__ import print_function

import numpy as np

import time
import pickle
import config   # training configs
cnn_config = config.cnn_config

import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

from nets import CNN, LeNet5, VGG16, ResNet152

np_output_file = 'exp_mnist_cnn_{}'.format(cnn_config['context'])

if __name__ == '__main__':
    train, test = datasets.get_mnist()
    train = [(np.reshape(data[0], (1, 28, 28)), data[1]) for data in train]
    test  = [(np.reshape(data[0], (1, 28, 28)), data[1]) for data in test]
    print('mnist Data loaded')
    print('train_data, len:{}, shape:{}'.format(len(train), train[0][0].shape))
    print('test_data, len:{}, shape:{}'.format(len(test), test[0][0].shape))
    print(type(train[0][0]), type(train[0][1]))

    # logging training time
    times = {}
    for _, net_config in enumerate(cnn_config['nets']):
        for b in cnn_config['batch_size']:
            batch_size = b
            train_iter = iterators.SerialIterator(train, batch_size=batch_size, shuffle=True)
            test_iter = iterators.SerialIterator(test, batch_size=batch_size, repeat=False, shuffle=False)

            key = "{}-batch{}".format(net_config['alias'], b)
            times[key] = []

            for t in range(cnn_config['test_times']):
                print("Current process: MLP {}, test {}".format(key, t))

                model = L.Classifier(CNN(net_config))
                if cnn_config['context'] == 'gpu':
                    model.to_gpu()
                optimizer = optimizers.Adam()
                optimizer.setup(model)

                train_iter.reset()
                test_iter.reset()

                if cnn_config['context'] == 'gpu':
                    updater = training.StandardUpdater(train_iter, optimizer, device=0)
                else:
                    updater = training.StandardUpdater(train_iter, optimizer)
                trainer = training.Trainer(updater, (cnn_config['epochs'], 'epoch'), out='result')
                    
                if cnn_config['context'] == 'gpu':
                    trainer.extend(extensions.Evaluator(test_iter, model, device=0))
                else:
                    trainer.extend(extensions.Evaluator(test_iter, model))
                trainer.extend(extensions.LogReport())
                trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
                trainer.extend(extensions.ProgressBar())

                ts = time.time()
                trainer.run()
                times[key].append(float(time.time()-ts))
    pickle.dump(times, open(np_output_file, 'wb'), True)
