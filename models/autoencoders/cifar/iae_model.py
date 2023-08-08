import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable


def iae_step(args, data, model_E, model_D, sampler, batch_idx, epoch, train_loader):
    # data.requires_grad_()
    z = model_E(data)
    X = model_D(z)
    # print(X.shape)
    # loss_R = (X - data).pow(2).mean()
    criterion = nn.MSELoss()
    loss_R = criterion(X, data)
    loss_W = torch.tensor(0)
    # loss_W = 0.09 * \
    #          (torch.norm(model_D.conv1.weight)
    #           + torch.norm(model_D.conv2.weight)
    #           + torch.norm(model_D.conv3.weight)
    #           + torch.norm(model_D.conv4.weight))


    u = torch.randn_like(z)
    u = u / u.norm(p=2, dim=1, keepdim=True)

    px = data.detach()
    px.requires_grad = True
    g_px = model_E(px)
    u_t_dg = autograd.grad(outputs=g_px, inputs=px,
                           grad_outputs=u,
                           create_graph=True, retain_graph=True, only_inputs=True)[0]
    u_t_dg_norm = u_t_dg.norm(p=2, dim=list(range(1, len(u_t_dg.shape))))
    loss_PISO = ((u_t_dg_norm - 1) ** 2).mean()

    pz = sampler.sample(z)
    pz.requires_grad_()

    f_pz = model_D(pz)
    v = torch.randn_like(f_pz)
    v.requires_grad_()

    u = torch.randn_like(pz)
    u = u / u.norm(p=2, dim=1, keepdim=True)

    v_t_df_pz = autograd.grad(outputs=f_pz, inputs=pz,
                              grad_outputs=v,  # torch.ones(f.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    df_pz_u = autograd.grad(outputs=v_t_df_pz, inputs=v,
                            grad_outputs=u,  # torch.ones(f.size()).to(device),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

    df_pz_u_norm = df_pz_u.norm(p=2, dim=list(range(1, len(df_pz_u.shape))))
    loss_ISO = ((df_pz_u_norm - 1) ** 2).mean()

    loss = loss_R + args.iso_weight * (loss_ISO + loss_PISO) + loss_W

    if batch_idx % args.log_interval == 0:
        print(
            'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss R: {:.12f}\tLoss ISO: {:.6f}\tLoss PISO: {:.6f}\tLoss W: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), loss_R.item(),
                loss_ISO.item(),
                loss_PISO.item(), loss_W.item()))

    return {'loss': loss, 'iso': loss_ISO, 'piso': loss_PISO, 'rec': loss_R, 'w': loss_W}

def simple_step(args, data, model_E, model_D, batch_idx, epoch, train_loader, train_len):
    # data = Variable(data.cuda())
    data.requires_grad_()
    z = model_E(data)

    X = model_D(z)

    loss_R = (X - data).pow(2).mean()
    # loss_D = torch.norm(X - data, p=2, dim=(1, 2, 3)).mean()

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


class Sampler:

    def sample(self, pnts):
        pnts_mean = pnts.mean(dim=0)
        pnts_std = pnts.std(dim=0)
        return torch.randn_like(pnts) * pnts_std + pnts_mean

class GaussianSampler(Sampler):

    def sample(self, pnts):
        pnts_mean = pnts.mean(dim=0)
        pnts_std = pnts.std(dim=0)
        return torch.randn_like(pnts) * pnts_std + pnts_mean

def weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
# Cnn Decoder

class SegNet_D(nn.Module):

    def __init__(self, args):
        super(SegNet_D, self).__init__()
        self.d_in = args.latent_dim

        nc = args.nc
        ndf = 128

        self.conv1 = nn.ConvTranspose2d(ndf * 8, ndf * 4, 4, 2, 1, bias=False)
        self.conv2 = nn.ConvTranspose2d(ndf * 4, ndf * 2, 4, 2, 1, bias=False)
        self.conv3 = nn.ConvTranspose2d(ndf * 2, ndf, 4, 2, 1, bias=False)
        self.conv4 = nn.ConvTranspose2d(ndf, nc, 4, 2, 1, bias=False)


        self.fc = nn.Linear(self.d_in, ndf * 8 * 2 * 2)
        self.main = nn.Sequential(
            self.conv1,
            nn.BatchNorm2d(ndf * 4),
            nn.Softplus(100),
            self.conv2,
            nn.BatchNorm2d(ndf * 2),
            nn.Softplus(100),
            self.conv3,
            nn.BatchNorm2d(ndf),
            nn.Softplus(100),
            self.conv4
        )

    def forward(self, x, layers=None):
        y = nn.Sigmoid()(self.fc(x))
        # y = y.unsqueeze(-1).unsqueeze(-1)
        y = y.reshape(-1, 1024, 2, 2)
        y = self.main(y)

        return y


# ===============================================================================================================
# Encoder
class SegNet_E(nn.Module):
    def __init__(self, args):
        super(SegNet_E, self).__init__()
        self.d_out = args.latent_dim

        nc = args.nc
        nef = 128
        self.main = nn.Sequential(
            # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, nef, 4, 2, 1, bias=False),
            nn.Softplus(100),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(nef, nef * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 2),
            nn.Softplus(100),
            # # state size. (ndf*2) x 8 x 8
            nn.Conv2d(nef * 2, nef * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 4),
            nn.Softplus(100),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(nef * 4, nef * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 8),
            nn.Softplus(100),
        )
        self.fc = nn.Linear(nef * 8 * 2 * 2, self.d_out)

    def forward(self, x):
        y = self.main(x).squeeze()
        # print(y.shape)
        y = y.reshape(-1, 4096)
        return self.fc(y)
        # return y


class Normalize(nn.Module):
    def __init__(self, mean, std, ndim=4, channels_axis=1, dtype=torch.float32):
        super(Normalize, self).__init__()
        shape = tuple(-1 if i == channels_axis else 1 for i in range(ndim))
        mean = torch.tensor(mean, dtype=dtype).reshape(shape)
        std = torch.tensor(std, dtype=dtype).reshape(shape)
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

    def forward(self, x):
        return (x.cuda() - self.mean.cuda()) / self.std.cuda()


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