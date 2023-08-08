import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import pickle
from datasets import dataset
import matplotlib.pyplot as plt
from views import basicview
# pca_dict = None


def load_PCA(pca_filename='pca.pickle', pca_path=None):
    # pca_dict = {"U": U, "S": S, "V": V}
    if pca_path is None:
        new_file = os.path.join('PCAs', pca_filename)
    else:
        new_file = pca_path
    with open(new_file, 'rb') as handle:
        pca_dict = pickle.load(handle)
    return pca_dict


class ProjectPCA(nn.Module):
    def __init__(self, k):
        super(ProjectPCA, self).__init__()
        self.k = torch.tensor(k)
        pca_dict = load_PCA()
        self.V = pca_dict.get("V")
        self.k = nn.Parameter(self.k, requires_grad=False)
        self.V = nn.Parameter(self.V, requires_grad=False)

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], 28*28))
        # zresult = torch.matmul(x, self.V[:, :self.k])
        zresult = torch.matmul(x, self.V)
        low_dim_x = torch.zeros_like(x)
        low_dim_x[:, :self.k] = 1
        low_dim_x = zresult * low_dim_x
        low_dim_x = torch.reshape(low_dim_x, (low_dim_x.shape[0], 1, 28, 28))
        return low_dim_x
        # print(zresult.shape, self.V[:, :self.k].shape)
        # xapprox = torch.matmul(self.V[:, :self.k], zresult.T)
        # # print(xapprox.shape)
        # xapprox = xapprox.T
        # img = torch.reshape(xapprox, (xapprox.shape[0], 1, 28, 28))
        # return img


class NormalizePCA(nn.Module):
    def __init__(self, k):
        super(NormalizePCA, self).__init__()
        train_loader = dataset.MNIST.getTrainSetIterator(batch_size=100)
        mean, std = batch_mean_and_sd(train_loader, k)
        self.mean = mean[0]
        self.std = std[0]
        self.mean = nn.Parameter(self.mean, requires_grad=False)
        self.std = nn.Parameter(self.std, requires_grad=False)

    def forward(self, x):
        x.view(-1)[:16] = (x.view(-1)[:16] - self.mean.cuda()) / self.std.cuda()
        # print(x)
        return x


def batch_mean_and_sd(loader, k):
    cnt = 0
    fst_moment = torch.empty(1)
    snd_moment = torch.empty(1)
    pca = ProjectPCA(k=k)

    for images, _ in loader:
        # print(images.shape)
        images = pca.forward(images)
        images = images.view(images.shape[0], -1)[:, :16]
        b, d = images.shape
        nb_pixels = b * d
        sum_ = torch.sum(images, dim=[0])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0])
        fst_moment = (cnt * fst_moment + sum_) / (
                cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (
                cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(
        snd_moment - fst_moment ** 2)
    return mean, std


if __name__ == '__main__':
    dic = load_PCA(pca_filename='pca32q10iters.pickle')
    print(dic.keys())
    print(dic["S"])
    plt.plot(dic["S"])
    plt.title("Singular values from PCA on MNIST")
    # plt.plot(50*torch.ones_like(dic["S"]))
    plt.show()
    # print('Scaled Mean Pixel Value {} \nScaled Pixel Values Std: {}'.format(train_loader.data.float().mean() / 255,
    #                                                                         train_loader.data.float().std() / 255))

    # print(res)