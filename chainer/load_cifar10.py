from __future__ import print_function

import numpy as np
import os

def unpickle(file):
    import pickle
    print(file)
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

cifar10_path = 'cifar-10-batches-py'

cifar10_training_data = []
cifar10_training_label = []
cifar10_testing_data = []
cifar10_testing_label = []
for f in os.listdir(cifar10_path):
    data = unpickle(os.path.join(cifar10_path, f))
    tmp = {}
    for key in data.keys():
        tmp[key.decode('ascii')] = data[key]
    if 'data_batch' in f:
        for d, l in zip(tmp['data'], tmp['labels']):
            d = np.reshape(d, (3, 32, 32))
            cifar10_training_data.append(d)
            cifar10_training_label.append(l)
    elif 'test' in f:
        for d, l in zip(tmp['data'], tmp['labels']):
            d = np.reshape(d, (3, 32, 32))
            cifar10_testing_data.append(d)
            cifar10_testing_label.append(l)
    else:
        pass

np.save('cifar10_training_data', cifar10_training_data)
np.save('cifar10_training_label', cifar10_training_label)
np.save('cifar10_testing_data', cifar10_testing_data)
np.save('cifar10_testing_label', cifar10_testing_label)

print('cifar10_training_data.shape', np.array(cifar10_training_data).shape)
print('cifar10_training_label.shape', np.array(cifar10_training_label).shape)
print('cifar10_testing_data.shape', np.array(cifar10_testing_data).shape)
print('cifar10_testing_label.shape', np.array(cifar10_testing_label).shape)

# a = cifar10_data[1]['data'][0]
# a = np.reshape(a, (3, 32, 32))
# print(type(a))
# print(a.shape)

# import scipy.misc
# scipy.misc.imsave('outfile.jpg', a)
# scipy.misc.toimage(a, cmin=0.0, cmax=...).save('outfile.jpg')

#pickle.dump(cifar10_data, open('cifar10_data', 'wb'), True)
