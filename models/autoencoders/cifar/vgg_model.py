import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
# from torchsummary import summary

# from config import device, imsize


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

    # loss_R = torch.norm(model_D.conv1.weight, p=2) + torch.norm(model_D.conv2.weight, p=2) + torch.norm(model_D.conv3.weight, p=2)

    # print(loss_D, loss_R)

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

# Cnn Decoder

class conv2DBatchNormRelu(nn.Module):
    def __init__(
            self,
            in_channels,
            n_filters,
            k_size,
            stride,
            padding,
            bias=True,
            dilation=1,
            with_bn=True,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels),
                             int(n_filters),
                             kernel_size=k_size,
                             padding=padding,
                             stride=stride,
                             bias=bias,
                             dilation=dilation, )

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          nn.ReLU(inplace=True))
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

class conv2DTransposeBatchNormRelu(nn.Module):
    def __init__(
            self,
            in_channels,
            n_filters,
            k_size,
            stride,
            padding,
            bias=True,
            dilation=1,
            with_bn=True,
    ):
        super(conv2DTransposeBatchNormRelu, self).__init__()

        conv_mod = nn.ConvTranspose2d(int(in_channels),
                             int(n_filters),
                             kernel_size=k_size,
                             padding=padding,
                             stride=stride,
                             bias=bias,
                             dilation=dilation, )

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          nn.ReLU(inplace=True))
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

class segnetDown2(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetDown3(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown3, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetUp2(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp2, self).__init__()
        # self.unpool = nn.MaxUnpool2d(2, 2)
        self.unpool = nn.Upsample(scale_factor=2)
        # self.unpool = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs):#, indices, output_shape):
        # outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.unpool(inputs)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class segnetUp3(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp3, self).__init__()
        # self.unpool = nn.MaxUnpool2d(2, 2)
        self.unpool = nn.Upsample(scale_factor=2)
        # self.unpool = conv2DTransposeBatchNormRelu(in_size, in_size, 4, 2, 1)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs):#, indices, output_shape):
        # print("segnetUp3", inputs.shape)
        # outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.unpool(inputs)
        outputs = self.conv1(outputs)
        # print("segnetUp3-after conv1", outputs.shape)
        outputs = self.conv2(outputs)
        # print("segnetUp3-after conv2", outputs.shape)
        outputs = self.conv3(outputs)
        # print("segnetUp3-after conv3", outputs.shape)
        return outputs

class SegNet_E(nn.Module):
    def __init__(self, args, in_channels=3, is_unpooling=True):
        super(SegNet_E, self).__init__()
        self.d_out = args.latent_dim
        self.in_channels = in_channels
        self.is_unpooling = is_unpooling

        self.down1 = segnetDown2(self.in_channels, 64)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown3(128, 256)
        self.down4 = segnetDown3(256, 512)
        self.down5 = segnetDown3(512, 512)
        self.down6 = nn.Linear(512, self.d_out)
        # print("encoders modules:", [x for x in self.modules()])
        self._initialize_weights()

    def forward(self, inputs):
        down1, indices_1, unpool_shape1 = self.down1(inputs)
        # print(down1.shape)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        # print(down2.shape)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        # print(down3.shape)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        # print(down4.shape)
        down5, indices_5, unpool_shape5 = self.down5(down4)
        # print(down5.shape)
        down5 = down5.reshape(-1, 512)
        down6 = self.down6(down5)
        # print(down6.shape)
        # print(unpool_shape1, unpool_shape2, unpool_shape3, unpool_shape4, unpool_shape5)
        # up5 = self.up5(down5, indices_5, unpool_shape5)
        # up4 = self.up4(up5, indices_4, unpool_shape4)
        # up3 = self.up3(up4, indices_3, unpool_shape3)
        # up2 = self.up2(up3, indices_2, unpool_shape2)
        # up1 = self.up1(up2, indices_1, unpool_shape1)

        indices = [indices_1, indices_2, indices_3, indices_4, indices_5]
        return down6#, indices

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

    def init_vgg16_params(self, vgg16):
        blocks = [self.down1, self.down2, self.down3, self.down4, self.down5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        vgg_layers = []
        for _layer in features:
            if isinstance(_layer, nn.Conv2d):
                vgg_layers.append(_layer)

        merged_layers = []
        for idx, conv_block in enumerate(blocks):
            if idx < 2:
                units = [conv_block.conv1.cbr_unit, conv_block.conv2.cbr_unit]
            else:
                units = [
                    conv_block.conv1.cbr_unit,
                    conv_block.conv2.cbr_unit,
                    conv_block.conv3.cbr_unit,
                ]
            for _unit in units:
                for _layer in _unit:
                    if isinstance(_layer, nn.Conv2d):
                        merged_layers.append(_layer)

        assert len(vgg_layers) == len(merged_layers)

        for l1, l2 in zip(vgg_layers, merged_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data

class SegNet_D(nn.Module):
    def __init__(self, args, in_channels=3, is_unpooling=True):
        super(SegNet_D, self).__init__()

        self.d_in = args.latent_dim
        self.in_channels = in_channels
        self.is_unpooling = is_unpooling

        # self.down1 = segnetDown2(self.in_channels, 64)
        # self.down2 = segnetDown2(64, 128)
        # self.down3 = segnetDown3(128, 256)
        # self.down4 = segnetDown3(256, 512)
        # self.down5 = segnetDown3(512, 512)
        self.up6 = nn.Linear(self.d_in, 512)
        self.up5 = segnetUp3(512, 512)
        self.up4 = segnetUp3(512, 256)
        self.up3 = segnetUp3(256, 128)
        self.up2 = segnetUp2(128, 64)
        self.up1 = segnetUp2(64, 3)
        # print("decoders modules:", [x for x in self.modules()])
        self._initialize_weights()

    def forward(self, inputs):#, indices):
        unpool_shape = [torch.Size([100, 64, 32, 32]), torch.Size([100, 128, 16, 16]), torch.Size([100, 256, 8, 8]),
                        torch.Size([100, 512, 4, 4]), torch.Size([100, 512, 2, 2])]
        # down1, indices_1, unpool_shape1 = self.down1(inputs)
        # down2, indices_2, unpool_shape2 = self.down2(down1)
        # down3, indices_3, unpool_shape3 = self.down3(down2)
        # down4, indices_4, unpool_shape4 = self.down4(down3)
        # down5, indices_5, unpool_shape5 = self.down5(down4)
        up6 = self.up6(inputs)
        up6 = up6.reshape(-1, 512, 1, 1)
        # print(indices[4].shape, indices[4].max(), indices[4].min())
        # print(indices[3].shape, indices[3].max(), indices[3].min())
        # print(indices[2].shape, indices[2].max(), indices[2].min())
        # print(indices[1].shape, indices[1].max(), indices[1].min())
        # print(indices[0].shape, indices[0].max(), indices[0].min())
        up5 = self.up5(up6)#, indices[4], unpool_shape[4])
        # print("up5.shape", up5.shape)
        up4 = self.up4(up5)#, indices[3], unpool_shape[3])
        # print("up4.shape", up4.shape)
        up3 = self.up3(up4)#, indices[2], unpool_shape[2])
        # print("up3.shape", up3.shape)
        up2 = self.up2(up3)#, indices[1], unpool_shape[1])
        # print("up2.shape", up2.shape)
        up1 = self.up1(up2)#, indices[0], unpool_shape[0])
        # print("up1.shape", up1.shape)

        return up1

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