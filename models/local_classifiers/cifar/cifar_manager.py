from models import utils
import os
import torchvision
from datasets import dataset
import torch.nn.functional as F
from models.local_classifiers.cifar.resnext import CifarResNeXt
from models.local_classifiers.cifar.simple_classifier import simple_cifar10


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn


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


class CifarManager:
    def __init__(self, model_name="vgg16", exp_name="cifar-sgd", i=-1):
        checkpoint_dir = utils.concat_home_dir(
            os.path.join('classifier', 'cifar10', model_name, exp_name, 'checkpoints'))
        print("checkpoint dir", checkpoint_dir)
        if model_name == "resnext":
            self.model = CifarResNeXt(
                cardinality=8,
                num_classes=10,
                depth=29,
                widen_factor=4,
                dropRate=0,
            )
            self.model = nn.Sequential(Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), self.model)

        elif model_name == "vgg16":
            self.model = torchvision.models.vgg16_bn()
            input_lastLayer = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(input_lastLayer, 10)

        elif model_name == "simple":
            self.model = simple_cifar10(128)
            self.model = nn.Sequential(Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), self.model)

        self.model = nn.DataParallel(self.model)
        self.model = self.model.cuda()

        dir = [int(i.split('_')[-1][:-3]) for i in os.listdir(checkpoint_dir)]
        if len(dir) > 0:
            print(dir)
            start_epoch = sorted(dir)[i]
            print("last epoch is {}".format(start_epoch))
            check_pt = torch.load(os.path.join(checkpoint_dir, 'epoch_{}.pt'.format(start_epoch)))
            self.model.load_state_dict(check_pt['model_state_dict'])

        if model_name == "vgg16":
            self.model = nn.Sequential(Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), self.model)
        self.model = self.model.eval()
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def test(self):
        test_loader = dataset.CIFARData.getTestSetIterator(200)
        counter = 0
        acc = 0

        for data, target in test_loader:
            with torch.no_grad():
                data, target = data.cuda(), target.cuda()
                counter += data.shape[0]
                correct = self.model(data).data.max(1)[1].eq(target)
                acc += correct.float().sum()

        print("total images tested: ", counter)
        print("correctly classified {}/{} ({:.0f}%)".format(acc, counter, 100 * (acc / counter)))

    def runModelOnImage(self, image, logging=True, withprobability=False):
        if len(image.shape) == 3:
            image = image.unsqueeze(0) # unsqueeze to add artificial first dimension

        output = self.model(image)
        # print("output", output)
        pred = output.data.max(1)[1]
        # print("pred", pred)
        predclass= self.classes[pred.item()]

        probs = F.softmax(output, dim=1)
        # print("probs", probs)
        probability = probs[0][pred.item()].item()

        if logging:
            resstr = "output: " + str(output) + "\npred: " + str(pred) + "\nclass: " + predclass +", with probability: " + str(probability)
            print("Running cifar on image: ", resstr)

        if withprobability:
            return output, predclass, probability

        return output, predclass

def load_model(model_name="vgg16", exp_name=None):
    if exp_name is not None:
        print("loading model", model_name)
        return CifarManager(model_name=model_name, exp_name=exp_name)

if __name__ == '__main__':
    manager = CifarManager(exp_name="cifar-sgd-new-nopretrain260-3", i=-1)
    manager.test_model()