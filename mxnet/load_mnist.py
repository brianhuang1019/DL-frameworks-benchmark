import mxnet as mx
import numpy as np

if __name__ == '__main__':
    mnist = mx.test_utils.get_mnist()

    print('MNIST Data loaded')
    print('train_data', mnist['train_data'].shape)
    print('train_label', mnist['train_label'].shape)
    print('test_data', mnist['test_data'].shape)
    print('test_label', mnist['test_label'].shape)

    np.save('mnist_training_data', mnist['train_data'])
    np.save('mnist_training_label', mnist['train_label'])
    np.save('mnist_testing_data', mnist['test_data'])
    np.save('mnist_testing_label', mnist['test_label'])