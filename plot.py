from __future__ import print_function

import pickle
import os
import re

import matplotlib as plt

# get exp file path
exp_dir = ['mxnet/exp', 'chainer/exp']
exp_file = []

def walk_path(d):
    for dirPath, dirNames, fileNames in os.walk(d):
        valid_fileNames = (f for f in fileNames if f != '.DS_Store')
        for f in valid_fileNames:
            file_path = os.path.join(dirPath, f)
            print(file_path)
            exp_file.append(parse_path(file_path))

def parse_path(fp):
    path_split = fp.split('/')

    framework = path_split[0]

    epoch = re.search('\d+', path_split[2].split('_')[0]).group()
    trail = re.search('\d+', path_split[2].split('_')[1]).group()

    dataset = path_split[3].split('_')[1]
    model = path_split[3].split('_')[2]
    context = path_split[3].split('_')[3]

    return {
        'file_path': fp,
        'framework': framework,
        'epoch': epoch,
        'trail': trail,
        'dataset': dataset,
        'model': model,
        'context': context
    }

if __name__ == '__main__':
    for d in exp_dir:
        walk_path(d)

    for exp in exp_file[:1]:
        exp_result = pickle.load(open(exp['file_path'], "rb"))
        if exp['model'] == 'cnn':
            pass
        elif exp['model'] == 'mlp':
            pass
        else:
            pass