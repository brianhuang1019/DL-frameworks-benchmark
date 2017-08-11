from __future__ import print_function

import pickle
import os
import re

import numpy as np
import matplotlib.pyplot as plt

import config
mlp_config = config.mlp_config
cnn_config = config.cnn_config

# get exp file path
exp_dir = ['mxnet/exp', 'chainer/exp']
exp_file = []

# exp settings
setting = 'epoch10-trail5'

# plot parameters
plot_type = 'save'
root_path = 'exp_img'
save_path = os.path.join(root_path, setting)
fixed1_path = os.path.join(save_path, 'fixed1')
fixed2_path = os.path.join(save_path, 'fixed2')
for path in [root_path, save_path, fixed1_path, fixed2_path]:
    if not os.path.exists(path):
        os.makedirs(path)

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

    epoch = re.search('\d+', path_split[2].split('-')[0]).group()
    trail = re.search('\d+', path_split[2].split('-')[1]).group()

    dataset = path_split[3].split('_')[1]
    model = path_split[3].split('_')[2]
    context = path_split[3].split('_')[3]

    return {
        'file_path': fp,
        'framework': framework,
        'epoch': int(epoch),
        'trail': int(trail),
        'dataset': dataset,
        'model': model,
        'context': context
    }

def statistic(data, m_type):
    if m_type == 'mlp':
        for key in data.keys():
            sec = np.array(data[key])
            mean = np.mean(sec)
            std = np.std(sec)
            data[key] = {
                'sec': sec,
                'mean': mean,
                'std': std,
                'layer': int(re.search('\d+', key.split('-')[0]).group()),
                'neuron': int(re.search('\d+', key.split('-')[1]).group()),
                'batch': int(re.search('\d+', key.split('-')[2]).group())
            }
    elif m_type == 'cnn':
        for key in data.keys():
            sec = np.array(data[key])
            mean = np.mean(sec)
            std = np.std(sec)
            data[key] = {
                'sec': sec,
                'mean': mean,
                'std': std,
                'alias': key.split('-')[0],
                'batch': int(re.search('\d+', key.split('-')[1]).group())
            }
    else:
        pass

def sort_keys(keys, sortby):
    assert len(sortby) <= 2

    ret = []
    for key in keys:
        ret.append({
            'key': key,
            'layer': int(re.search('\d+', key.split('-')[0]).group()),
            'neuron': int(re.search('\d+', key.split('-')[1]).group()),
            'batch': int(re.search('\d+', key.split('-')[2]).group())
        })

    if len(sortby) == 2:
        ret = sorted(ret, key = lambda x: (x[sortby[0]], x[sortby[1]]))
    else:
        ret = sorted(ret, key = lambda x: x[sortby[0]])
    return ret

def get_val_from_keys(keys, dataset1, dataset2):
    m_means = []
    m_std = []
    c_means = []
    c_std = []
    ticks = []
    for key in keys:
        ticks.append(key)
        m_means.append(dataset1[key]['mean'])
        m_std.append(dataset1[key]['std'])
        c_means.append(dataset2[key]['mean'])
        c_std.append(dataset2[key]['std'])

    return m_means, m_std, c_means, c_std, ticks

def plot(m1, s1, m2, s2, n_groups, ticks=None, title=None, bar_width=0.4, opacity=0.4, p_type='show', save_path=save_path):
        
    fig, ax = plt.subplots(figsize=(13, 8), dpi=85)
    
    index = np.arange(n_groups)

    error_config = {'ecolor': '0.3'}
    
    rects1 = plt.barh(index, m_means, bar_width,
                     alpha=opacity,
                     color='b',
                     xerr=m_std,
                     error_kw=error_config,
                     label='MXNet')
    
    rects2 = plt.barh(index + bar_width, c_means, bar_width,
                     alpha=opacity,
                     color='r',
                     xerr=c_std,
                     error_kw=error_config,
                     label='Chainer')
    
    plt.xlabel('Sec(s)')
    plt.ylabel('model configs')
    plt.yticks(index + bar_width / 2, ticks)
    plt.title(title)
    plt.legend()
    
    plt.tight_layout()
    if p_type == 'save':
        plt.savefig(os.path.join(save_path, title))
    else:
        plt.show()

if __name__ == '__main__':
    for d in exp_dir:
        walk_path(d)

    m_mnist_mlp = pickle.load(open('mxnet/exp/{}/exp_mnist_mlp_gpu'.format(setting), 'rb'))
    c_mnist_mlp = pickle.load(open('chainer/exp/{}/exp_mnist_mlp_gpu'.format(setting), 'rb'))
    m_mnist_cnn = pickle.load(open('mxnet/exp/{}/exp_mnist_cnn_gpu'.format(setting), 'rb'))
    c_mnist_cnn = pickle.load(open('chainer/exp/{}/exp_mnist_cnn_gpu'.format(setting), 'rb'))
    m_cifar10_mlp = pickle.load(open('mxnet/exp/{}/exp_cifar10_mlp_gpu'.format(setting), 'rb'))
    c_cifar10_mlp = pickle.load(open('chainer/exp/{}/exp_cifar10_mlp_gpu'.format(setting), 'rb'))
    m_cifar10_cnn = pickle.load(open('mxnet/exp/{}/exp_cifar10_cnn_gpu'.format(setting), 'rb'))
    c_cifar10_cnn = pickle.load(open('chainer/exp/{}/exp_cifar10_cnn_gpu'.format(setting), 'rb'))

    statistic(m_mnist_mlp, m_type='mlp')
    statistic(c_mnist_mlp, m_type='mlp')
    statistic(m_mnist_cnn, m_type='cnn')
    statistic(c_mnist_cnn, m_type='cnn')
    statistic(m_cifar10_mlp, m_type='mlp')
    statistic(c_cifar10_mlp, m_type='mlp')
    statistic(m_cifar10_cnn, m_type='cnn')
    statistic(c_cifar10_cnn, m_type='cnn')

    mlp_exp_groups = [{'member1':m_mnist_mlp, 'member2':c_mnist_mlp, 'name':'mnist_mlp', 'setting':setting}, {'member1':m_cifar10_mlp, 'member2':c_cifar10_mlp, 'name':'cifar10_mlp', 'setting':setting}]
    cnn_exp_groups = [{'member1':m_mnist_cnn, 'member2':c_mnist_cnn, 'name':'mnist_cnn', 'setting':setting}, {'member1':m_cifar10_cnn, 'member2':c_cifar10_cnn, 'name':'cifar10_cnn', 'setting':setting}]

    # fixed layer
    for layer in mlp_config['layers']:
        for group in mlp_exp_groups:
            dict_keys = sort_keys([key for key in group['member1'].keys() if group['member1'][key]['layer'] == layer], sortby=['neuron', 'batch'])
    
            keys = [x['key'] for x in dict_keys]
            m_means, m_std, c_means, c_std, ticks = get_val_from_keys(keys, group['member1'], group['member2'])
    
            n_groups = len(mlp_config['neurons']) * len(mlp_config['batch_size'])
            if plot_type == 'save':
                plot(m1=m_means, s1=m_std, m2=c_means, s2=c_std, ticks=ticks, n_groups=n_groups, p_type='save', title='{}_{}_layer{}'.format(group['name'], group['setting'], layer), save_path=fixed1_path)
            else:
                plot(m1=m_means, s1=m_std, m2=c_means, s2=c_std, ticks=ticks, n_groups=n_groups, p_type='show', title='{}_{}_layer{}'.format(group['name'], group['setting'], layer))
            print('Finish {}_{}_layer{}'.format(group['name'], group['setting'], layer))

    # fixed neuron
    for neuron in mlp_config['neurons']:
        for group in mlp_exp_groups:
            dict_keys = sort_keys([key for key in group['member1'].keys() if group['member1'][key]['neuron'] == neuron], sortby=['layer', 'batch'])

            keys = [x['key'] for x in dict_keys] 
            m_means, m_std, c_means, c_std, ticks = get_val_from_keys(keys, group['member1'], group['member2'])
    
            n_groups = len(mlp_config['layers']) * len(mlp_config['batch_size'])
            if plot_type == 'save':
                plot(m1=m_means, s1=m_std, m2=c_means, s2=c_std, ticks=ticks, n_groups=n_groups, p_type='save', title='{}_{}_neuron{}'.format(group['name'], group['setting'], neuron), save_path=fixed1_path)
            else:
                plot(m1=m_means, s1=m_std, m2=c_means, s2=c_std, ticks=ticks, n_groups=n_groups, p_type='show', title='{}_{}_neuron{}'.format(group['name'], group['setting'], neuron))
            print('Finish {}_{}_neuron{}'.format(group['name'], group['setting'], neuron))

    # fixed batch
    for batch in mlp_config['batch_size']:
        for group in mlp_exp_groups:
            dict_keys = sort_keys([key for key in group['member1'].keys() if group['member1'][key]['batch'] == batch], sortby=['layer', 'neuron'])

            keys = [x['key'] for x in dict_keys]     
            m_means, m_std, c_means, c_std, ticks = get_val_from_keys(keys, group['member1'], group['member2'])
    
            n_groups = len(mlp_config['layers']) * len(mlp_config['neurons'])
            if plot_type == 'save':
                plot(m1=m_means, s1=m_std, m2=c_means, s2=c_std, ticks=ticks, n_groups=n_groups, p_type='save', title='{}_{}_batch{}'.format(group['name'], group['setting'], batch), save_path=fixed1_path)
            else:
                plot(m1=m_means, s1=m_std, m2=c_means, s2=c_std, ticks=ticks, n_groups=n_groups, p_type='show', title='{}_{}_batch{}'.format(group['name'], group['setting'], batch))
            print('Finish {}_{}_batch{}'.format(group['name'], group['setting'], batch))

    # fixed layer, neuron
    for layer in mlp_config['layers']:
        for neuron in mlp_config['neurons']:
            for group in mlp_exp_groups:
                dict_keys = sort_keys([key for key in group['member1'].keys() if group['member1'][key]['layer'] == layer and group['member1'][key]['neuron'] == neuron], sortby=['batch'])
        
                keys = [x['key'] for x in dict_keys]
                m_means, m_std, c_means, c_std, ticks = get_val_from_keys(keys, group['member1'], group['member2'])
        
                n_groups = len(mlp_config['batch_size'])
                if plot_type == 'save':
                    plot(m1=m_means, s1=m_std, m2=c_means, s2=c_std, ticks=ticks, n_groups=n_groups, p_type='save', title='{}_{}_layer{}-neuron{}'.format(group['name'], group['setting'], layer, neuron), save_path=fixed2_path)
                else:
                    plot(m1=m_means, s1=m_std, m2=c_means, s2=c_std, ticks=ticks, n_groups=n_groups, p_type='show', title='{}_{}_layer{}-neuron{}'.format(group['name'], group['setting'], layer, neuron))
                print('Finish {}_{}_layer{}'.format(group['name'], group['setting'], layer))

    # fixed layer, batch
    for layer in mlp_config['layers']:
        for batch in mlp_config['batch_size']:
            for group in mlp_exp_groups:
                dict_keys = sort_keys([key for key in group['member1'].keys() if group['member1'][key]['layer'] == layer and group['member1'][key]['batch'] == batch], sortby=['neuron'])
        
                keys = [x['key'] for x in dict_keys]
                m_means, m_std, c_means, c_std, ticks = get_val_from_keys(keys, group['member1'], group['member2'])
        
                n_groups = len(mlp_config['neurons'])
                if plot_type == 'save':
                    plot(m1=m_means, s1=m_std, m2=c_means, s2=c_std, ticks=ticks, n_groups=n_groups, p_type='save', title='{}_{}_layer{}-batch{}'.format(group['name'], group['setting'], layer, batch), save_path=fixed2_path)
                else:
                    plot(m1=m_means, s1=m_std, m2=c_means, s2=c_std, ticks=ticks, n_groups=n_groups, p_type='show', title='{}_{}_layer{}-batch{}'.format(group['name'], group['setting'], layer, batch))
                print('Finish {}_{}_layer{}'.format(group['name'], group['setting'], layer))

    # fixed neuron, batch
    for neuron in mlp_config['neurons']:
        for batch in mlp_config['batch_size']:
            for group in mlp_exp_groups:
                dict_keys = sort_keys([key for key in group['member1'].keys() if group['member1'][key]['neuron'] == neuron and group['member1'][key]['batch'] == batch], sortby=['layer'])
        
                keys = [x['key'] for x in dict_keys]
                m_means, m_std, c_means, c_std, ticks = get_val_from_keys(keys, group['member1'], group['member2'])
        
                n_groups = len(mlp_config['layers'])
                if plot_type == 'save':
                    plot(m1=m_means, s1=m_std, m2=c_means, s2=c_std, ticks=ticks, n_groups=n_groups, p_type='save', title='{}_{}_neuron{}-batch{}'.format(group['name'], group['setting'], neuron, batch), save_path=fixed2_path)
                else:
                    plot(m1=m_means, s1=m_std, m2=c_means, s2=c_std, ticks=ticks, n_groups=n_groups, p_type='show', title='{}_{}_neuron{}-batch{}'.format(group['name'], group['setting'], neuron, batch))
                print('Finish {}_{}_layer{}'.format(group['name'], group['setting'], layer))

    # plot CNN
    for group in cnn_exp_groups:
        dict_keys = []
        for key in group['member1'].keys():
            dict_keys.append({
                'key': key,
                'alias': key.split('-')[0],
                'batch': int(re.search('\d+', key.split('-')[1]).group())
            })
        dict_keys = sorted(dict_keys, key = lambda x: x['batch'])

        keys = [x['key'] for x in dict_keys]
        m_means, m_std, c_means, c_std, ticks = get_val_from_keys(keys, group['member1'], group['member2'])

        if plot_type == 'save':       
            plot(m1=m_means, s1=m_std, m2=c_means, s2=c_std, ticks=ticks, n_groups=len(cnn_config['batch_size']), p_type='save', title='{}_{}'.format(group['name'], group['setting']))
        else:
            plot(m1=m_means, s1=m_std, m2=c_means, s2=c_std, ticks=ticks, n_groups=len(cnn_config['batch_size']), p_type='show', title='{}_{}'.format(group['name'], group['setting']))
        print('Finish {}_{}'.format(group['name'], group['setting']))

