from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from IPython import embed


def simple_step(args, data, model_E, model_D, batch_idx, epoch, train_loader, train_len):
    # data = Variable(data.cuda())
    data.requires_grad_()
    # [z, indices, unpoolshape] = model_E(data)
    z = model_E(data)
    # indices_r = [torch.randint_like(indices[i], high=2 ** (10 - 2 * i)).long() for i in range(len(indices))]
    # print(z.shape)
    X = model_D(z)
    # print(data.shape, z.shape, X.shape)

    # loss_R =  (X - data).pow(2).mean()
    loss_R = torch.norm(X - data, p=2, dim=(1, 2, 3)).mean()

    # criterion = nn.BCELoss()
    # loss = criterion(data, Variable(X))
    # loss = loss_D + 0.08 * loss_R
    loss = loss_R


    if batch_idx % args.log_interval == 0:
        print(
            'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss R: {:.12f}'.format(
                epoch, batch_idx * len(data), train_len,
                       100. * batch_idx * len(data) / train_len, loss.item(), loss_R.item()))

    return {'loss': loss, 'rec': loss_R}


class SegNet_E(nn.Module):
    def __init__(self, args, in_channels=3, is_unpooling=True):
        super(SegNet_E, self).__init__()
        self.d_out = args.latent_dim
        self.in_channels = in_channels
        self.is_unpooling = is_unpooling

        n_channel = 128
        cfg = [n_channel, n_channel, 'M', 2 * n_channel, 2 * n_channel, 'M', 4 * n_channel, 4 * n_channel, 'M',
               (8 * n_channel, 0), 'M']
        self.features = make_layers(cfg, batch_norm=True)

        linear = nn.Linear(n_channel*8, self.d_out)
        # linear.weight.data = linear.weight.data.type(torch.FloatTensor).cuda()
        # linear.bias.data = linear.bias.data.type(torch.FloatTensor).cuda()
        self.classifier = nn.Sequential(linear)
        # print(self.features)
        # print(self.classifier)
        # self._initialize_weights()
        # print("encoders modules:", [x for x in self.modules()])

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class SegNet_D(nn.Module):
    def __init__(self, args, in_channels=3, is_unpooling=True):
        super(SegNet_D, self).__init__()
        self.d_in = args.latent_dim
        self.in_channels = in_channels
        self.is_unpooling = is_unpooling

        self.n_channel = 128
        cfg = [3, self.n_channel, 'U', self.n_channel, 2 * self.n_channel, 'U', 2 * self.n_channel, 4 * self.n_channel, 'U', (4 * self.n_channel, 0), 'U']
        self.features = make_layers(cfg[::-1], batch_norm=True, in_channels=self.n_channel*8, transpose=True)

        linear = nn.Linear(self.d_in, self.n_channel*8)
        # linear.weight.data = linear.weight.data.type(torch.FloatTensor).cuda()
        # linear.bias.data = linear.bias.data.type(torch.FloatTensor).cuda()
        self.linear = nn.Sequential(linear)
        # print(self.features)
        # print(self.classifier)
        # self._initialize_weights()
        # print("decoders modules:", [x for x in self.modules()])

    def forward(self, x):
        # print("-", x.shape)
        x = self.linear(x)
        # print(x.shape)
        x = x.reshape(-1, self.n_channel * 8, 1, 1)
        # print(x.shape)
        x = self.features(x)
        # print(x.shape)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Normalize(nn.Module):
    def __init__(self, mean, std, ndim=4, channels_axis=1):
        super(Normalize, self).__init__()
        shape = tuple(-1 if i == channels_axis else 1 for i in range(ndim))
        mean = torch.tensor(mean).reshape(shape)
        std = torch.tensor(std).reshape(shape)
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

    def forward(self, x):
        return (x.cuda() - self.mean.cuda()) / self.std.cuda()


class DeNormalize(nn.Module):
    def __init__(self, mean, std, ndim=4, channels_axis=1):
        super(DeNormalize, self).__init__()
        shape = tuple(-1 if i == channels_axis else 1 for i in range(ndim))
        mean = torch.tensor(mean).reshape(shape)
        std = torch.tensor(std).reshape(shape)
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

    def forward(self, x):
        return (x.cuda() * self.std.cuda()) - self.mean.cuda()


def make_layers(cfg, batch_norm=False, in_channels=3, transpose=False):
    layers = []
    # in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'U':
            layers += [nn.Upsample(scale_factor=2)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            if transpose:
                conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=padding)
            else:
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



