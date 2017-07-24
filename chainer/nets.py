import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class MLP(Chain):
    def __init__(self, n_layers, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l_first = L.Linear(None, n_units)  # n_in -> n_units
            self.n_layers = n_layers
            for idx in range(self.n_layers-2):
                exec("self.l_inter_{} = L.Linear(n_units, n_units)".format(idx)) # n_units -> n_units
            self.l_last = L.Linear(n_units, n_out)    # n_units -> n_out

    def __call__(self, x):
        h = F.relu(self.l_first(x))
        for idx in range(self.n_layers-2):
            exec("h = F.relu(self.l_inter_{}(h))".format(idx))
        y = self.l_last(h)
        return y

class Classifier(Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__()
        with self.init_scope():
            self.predictor = predictor
    
    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        report({'loss': loss, 'accuracy': accuracy}, self)
        return loss

class CNN(Chain):
    def __init__(self, n_config):
        super(CNN, self).__init__()

        print('net config', n_config)

        act_type = n_config['act']
        kernal = (n_config['kernal_v'], n_config['kernal_h'])
        stride = (n_config['stride_v'], n_config['stride_h'])
        self.depths = n_config['depths']
    
        pool_type = n_config['pool']['type']
        self.pool_kernal = (n_config['pool']['kernal_v'], n_config['pool']['kernal_h'])
        self.pool_stride = (n_config['pool']['stride_v'], n_config['pool']['stride_h'])
    
        self.fc_neurons = n_config['fc_neurons']

        with self.init_scope():
            self.conv_first = L.Convolution2D(
                in_channels=None, out_channels=self.depths[0], ksize=kernal, stride=stride)
            for idx, filters in enumerate(self.depths[1:]):
                exec("self.conv_inter_{} = L.Convolution2D(in_channels=self.depths[{}], out_channels={}, ksize={}, stride={})".format(idx, idx, filters, kernal, stride))
            for idx, neurons in enumerate(self.fc_neurons[:-1]):
                exec("self.fc_inter_{} = L.Linear(None, {})".format(idx, neurons))
            self.fc_last = L.Linear(None, self.fc_neurons[-1])

        if act_type == "sigmoid":
            self.act = F.sigmoid
        elif act_type == "tanh":
            self.act = F.tanh
        elif act_type == "relu":
            self.act = F.relu
        else:
            self.act = F.relu

        if pool_type == "max":
            self.pool = F.max_pooling_2d
        elif pool_type == "avg":
            self.pool = F.average_pooling_2d
        else:
            self.pool = F.max_pooling_2d

    def __call__(self, x):
        h = self.act(self.conv_first(x))
        h = self.pool(h, ksize=self.pool_kernal, stride=self.pool_stride)
        for idx, _ in enumerate(self.depths[1:]):
            exec("h = self.act(self.conv_inter_{}(h))".format(idx))
            h = self.pool(h, ksize=self.pool_kernal, stride=self.pool_stride)
        for idx, _ in enumerate(self.fc_neurons[:-1]):
            exec("h = self.act(self.fc_inter_{}(h))".format(idx))            
        if chainer.config.train:
            return self.fc_last(h)
        return F.softmax(self.fc_last(h))

class LeNet5(Chain):
    def __init__(self):
        super(LeNet5, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels=1, out_channels=6, ksize=5, stride=1)
            self.conv2 = L.Convolution2D(
                in_channels=6, out_channels=16, ksize=5, stride=1)
            self.conv3 = L.Convolution2D(
                in_channels=16, out_channels=120, ksize=4, stride=1)
            self.fc4 = L.Linear(None, 84)
            self.fc5 = L.Linear(84, 10)

    def __call__(self, x):
        h = F.sigmoid(self.conv1(x))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.sigmoid(self.conv2(h))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.sigmoid(self.conv3(h))
        h = F.sigmoid(self.fc4(h))
        if chainer.config.train:
            return self.fc5(h)
        return F.softmax(self.fc5(h))

class VGG16(chainer.ChainList):
    def __init__(self):
        w = chainer.initializers.HeNormal()
        super(VGG16, self).__init__(
            VGGBlock(64),
            VGGBlock(128),
            VGGBlock(256, 3),
            VGGBlock(512, 3),
            VGGBlock(512, 3, True))

    def __call__(self, x):
        for f in self.children():
            x = f(x)
        if chainer.config.train:
            return x
        return F.softmax(x)


class VGGBlock(chainer.Chain):
    def __init__(self, n_channels, n_convs=2, fc=False):
        w = chainer.initializers.HeNormal()
        super(VGGBlock, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, n_channels, 3, 1, 1, initialW=w)
            self.conv2 = L.Convolution2D(
                n_channels, n_channels, 3, 1, 1, initialW=w)
            if n_convs == 3:
                self.conv3 = L.Convolution2D(
                    n_channels, n_channels, 3, 1, 1, initialW=w)
            if fc:
                self.fc4 = L.Linear(None, 4096, initialW=w)
                self.fc5 = L.Linear(4096, 4096, initialW=w)
                self.fc6 = L.Linear(4096, 1000, initialW=w)

        self.n_convs = n_convs
        self.fc = fc

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        if self.n_convs == 3:
            h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(h, 2, 2)
        if self.fc:
            h = F.dropout(F.relu(self.fc4(h)))
            h = F.dropout(F.relu(self.fc5(h)))
            h = self.fc6(h)
        return h

class ResNet152(chainer.Chain):
    def __init__(self, n_blocks=[3, 8, 36, 3]):
        w = chainer.initializers.HeNormal()
        super(ResNet152, self).__init__(
            conv1=L.Convolution2D(
                None, 64, 7, 2, 3, initialW=w, nobias=True),
            bn1=L.BatchNormalization(64),
            res2=ResBlock(n_blocks[0], 64, 64, 256, 1),
            res3=ResBlock(n_blocks[1], 256, 128, 512),
            res4=ResBlock(n_blocks[2], 512, 256, 1024),
            res5=ResBlock(n_blocks[3], 1024, 512, 2048),
            fc6=L.Linear(2048, 1000))

    def __call__(self, x):
        h = self.bn1(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 2, 2)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = F.average_pooling_2d(h, h.shape[2:], stride=1)
        h = self.fc6(h)
        if chainer.config.train:
            return h
        return F.softmax(h)


class ResBlock(chainer.ChainList):
    def __init__(self, n_layers, n_in, n_mid, n_out, stride=2):
        w = chainer.initializers.HeNormal()
        super(ResBlock, self).__init__()
        self.add_link(BottleNeck(n_in, n_mid, n_out, stride, True))
        for _ in range(n_layers - 1):
            self.add_link(BottleNeck(n_out, n_mid, n_out))

    def __call__(self, x):
        for f in self.children():
            x = f(x)
        return x


class BottleNeck(chainer.Chain):
    def __init__(self, n_in, n_mid, n_out, stride=1, proj=False):
        w = chainer.initializers.HeNormal()
        super(BottleNeck, self).__init__()
        with self.init_scope():
            self.conv1x1a = L.Convolution2D(
                n_in, n_mid, 1, stride, 0, initialW=w, nobias=True)
            self.conv3x3b = L.Convolution2D(
                n_mid, n_mid, 3, 1, 1, initialW=w, nobias=True)
            self.conv1x1c = L.Convolution2D(
                n_mid, n_out, 1, 1, 0, initialW=w, nobias=True)
            self.bn_a = L.BatchNormalization(n_mid)
            self.bn_b = L.BatchNormalization(n_mid)
            self.bn_c = L.BatchNormalization(n_out)
            if proj:
                self.conv1x1r = L.Convolution2D(
                    n_in, n_out, 1, stride, 0, initialW=w, nobias=True)
                self.bn_r = L.BatchNormalization(n_out)
        self.proj = proj

    def __call__(self, x):
        h = F.relu(self.bn_a(self.conv1x1a(x)))
        h = F.relu(self.bn_b(self.conv3x3b(h)))
        h = self.bn_c(self.conv1x1c(h))
        if self.proj:
            x = self.bn_r(self.conv1x1r(x))
        return F.relu(h + x)