import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from classifiers.MNIST import model

from MNIST.PCAMNIST import ProjectPCA, NormalizePCA


class MLPnPCAClassifier(nn.Module):
    def __init__(self, num_of_dims):
        super(MLPnPCAClassifier, self).__init__()
        self.mlp = MLPClassifier(data_parallel=False, pretrained=None)
        self.classes = self.mlp.classes
        # self.model = self.mlp.model
        self.pca = ProjectPCA(k=num_of_dims)
        self.normalize = NormalizePCA(k=num_of_dims)

        # self.model = nn.Sequential(self.pca, self.normalize, self.mlp)
        # self.pre_process = nn.Sequential(self.pca, self.normalize)

    def pre_process(self, x):
        res = self.pca(x)
        # res = self.normalize(res)
        # data_res = res.view(res.shape[0], -1)[:, :16]
        # qn = torch.norm(data_res, p=2, dim=1).detach()
        # qn = qn.reshape(qn.shape[0], 1, 1, 1)
        # qn = qn.expand_as(res)
        # # print(qn.shape)
        # res = res.div(qn.expand_as(res))
        # res = np.sqrt(784) * res
        return res

    def forward(self, x):
        res = self.pre_process(x)
        return self.mlp.forward(res)

    def runModelOnImage(self, image, logging=True, withprobability=False):
        image = image.type(torch.FloatTensor).cuda()
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        output = self.model(image)[0]
        prob = output.softmax(0)
        probability, pred = prob.topk(k=1, dim=0)
        # print(pred)
        # print(probability)
        predclass = self.classes[pred.item()]

        if logging:
            resstr = "\npred: " + str(
                pred) + "\nclass: " + predclass + ", with probability: " + str(probability.item())
            print("Running cifar on image: ", resstr)

        if withprobability:
            return output, predclass, probability.item()

        return output, predclass


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


class MLPClassifier(nn.Module):
    def __init__(self, data_parallel=True, pretrained=False):
        super(MLPClassifier, self).__init__()
        # self.fc1 = nn.Linear(in_features=784, out_features=256)
        # self.relu1 = nn.ReLU()
        # self.drop1 = nn.Dropout(0.2)
        # self.fc2 = nn.Linear(in_features=256, out_features=256)
        # self.relu2 = nn.ReLU()
        # self.drop2 = nn.Dropout(0.2)
        # self.fc3 = nn.Linear(in_features=256, out_features=10)
        self.model = model.mnist(input_dims=784, n_hiddens=[256, 256], n_class=10, src="mnist")
        self.fc1 = self.model.model[0]
        # self.model = nn.Sequential(self.fc1, self.relu1, self.drop1, self.fc2, self.relu2, self.drop2, self.fc3)
        # self.model = nn.Sequential(self.fc1, self.relu1, self.fc2, self.relu2, self.fc3)
        self.normalize = Normalize(mean=(0.1307,), std=(0.3081,))
        # self.model = nn.Sequential(normalize, self.model)
        if data_parallel:
            self.model = torch.nn.DataParallel(self.model)

        self.model = self.model.cuda()
        self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    def forward(self, x):
        # print(x.shape)
        x = self.normalize(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        # x = self.fc1(x)
        # x = self.relu1(x)
        # x= self. drop1(x)
        # x = self.fc2(x)
        # x= self.relu2(x)
        # x = self.drop2(x)
        # x = self.fc3(x)
        # print(x.shape)
        return self.model.forward(x)
        # return x

    def runModelOnImage(self, image, logging=True, withprobability=False):
        image = image.type(torch.FloatTensor).cuda()
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        output = self.model(image)[0]
        prob = output.softmax(0)
        probability, pred = prob.topk(k=1, dim=0)
        # print(pred)
        # print(probability)
        predclass = self.classes[pred.item()]

        if logging:
            resstr = "\npred: " + str(
                pred) + "\nclass: " + predclass + ", with probability: " + str(probability.item())
            print("Running cifar on image: ", resstr)

        if withprobability:
            return output, predclass, probability.item()

        return output, predclass


class MLP1Classifier(nn.Module):
    def __init__(self, data_parallel=True, pretrained=True):
        super(MLP1Classifier, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=256)
        self.relu1 = nn.ReLU()
        # self.drop1 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(in_features=256, out_features=10)
        # self.fc3.requires_grad_(False)
        # self.model = model.mnist(input_dims=784, n_hiddens=[256, 256], n_class=10, pretrained=pretrained, src="mnist")
        self.model = nn.Sequential(self.fc1, self.relu1, self.fc3)
        # self.model = nn.Sequential(self.fc1, self.relu1, self.drop1, self.fc3)
        self.normalize = Normalize(mean=(0.1307,), std=(0.3081,))
        # self.model = nn.Sequential(normalize, self.model)
        if data_parallel:
            self.model = torch.nn.DataParallel(self.model)

        self.model = self.model.cuda()
        self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    def forward(self, x):
        # print(x.shape)
        x = self.normalize(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        # x = self.fc1(x)
        # x = self.relu1(x)
        # x= self. drop1(x)
        # x = self.fc2(x)
        # x= self.relu2(x)
        # x = self.drop2(x)
        # x = self.fc3(x)
        # print(x.shape)
        return self.model.forward(x)
        # return x

    def runModelOnImage(self, image, logging=True, withprobability=False):
        image = image.type(torch.FloatTensor).cuda()
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        output = self.model(image)[0]
        prob = output.softmax(0)
        probability, pred = prob.topk(k=1, dim=0)
        # print(pred)
        # print(probability)
        predclass = self.classes[pred.item()]

        if logging:
            resstr = "\npred: " + str(
                pred) + "\nclass: " + predclass + ", with probability: " + str(probability.item())
            print("Running cifar on image: ", resstr)

        if withprobability:
            return output, predclass, probability.item()

        return output, predclass