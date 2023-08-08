from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
# from IPython import embed

model_urls = {
    'cifar10': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/cifar10-d875770b.pth',
    'cifar100': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/cifar100-3a55a987.pth',
}


class CIFAR(nn.Module):
    def __init__(self, features, n_channel, num_classes):
        super(CIFAR, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        linear = nn.Linear(n_channel, num_classes)
        # linear.weight.data = linear.weight.data.type(torch.FloatTensor).cuda()
        # linear.bias.data = linear.bias.data.type(torch.FloatTensor).cuda()
        self.classifier = nn.Sequential(linear)
        # print(self.features)
        # print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
            # conv2d.weight.data = conv2d.weight.data.type(torch.FloatTensor).cuda()
            # conv2d.bias.data = conv2d.bias.data.type(torch.FloatTensor).cuda()

            if batch_norm:
                norm = nn.BatchNorm2d(out_channels, affine=False)
                # norm.running_mean.data = norm.running_mean.data.type(torch.FloatTensor).cuda()
                # norm.running_var.data = norm.running_var.data.type(torch.FloatTensor).cuda()
                layers += [conv2d, norm, nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = out_channels
    return nn.Sequential(*layers)


def simple_cifar10(n_channel, pretrained=None):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
    layers = make_layers(cfg, batch_norm=True)
    model = CIFAR(layers, n_channel=8*n_channel, num_classes=10)
    if pretrained is not None:
        m = model_zoo.load_url(model_urls['cifar10'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model

def cifar100(n_channel, pretrained=None):
    cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
    layers = make_layers(cfg, batch_norm=True)
    model = CIFAR(layers, n_channel=8*n_channel, num_classes=100)
    if pretrained is not None:
        m = model_zoo.load_url(model_urls['cifar100'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model


# class Normalize(nn.Module):
#     def __init__(self, mean, std, ndim=4, channels_axis=1, dtype=torch.float64):
#         super(Normalize, self).__init__()
#         shape = tuple(-1 if i == channels_axis else 1 for i in range(ndim))
#         mean = torch.tensor(mean, dtype=dtype).reshape(shape)
#         std = torch.tensor(std, dtype=dtype).reshape(shape)
#         self.mean = nn.Parameter(mean, requires_grad=False)
#         self.std = nn.Parameter(std, requires_grad=False)
#
#     def forward(self, x):
#         return (x.cuda() - self.mean.cuda()) / self.std.cuda()

# if __name__ == '__main__':
    # model = cifar10(128, pretrained='log/cifar10/best-135.pth')
    # embed()

