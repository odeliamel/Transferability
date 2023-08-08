import argparse
import torch
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from trained.cifar_deeperncoder.model import *
import models.utils as utils
from datasets import dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from torch.utils.tensorboard import SummaryWriter
from models.classifiers.cifar.resnext import CifarResNeXt
from models.classifiers.cifar.simple_classifier import simple_cifar10

from threadpoolctl import threadpool_limits
_thread_limit = threadpool_limits(limits=8)
print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


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


def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR10 resnext classifier')

    # experiment data
    parser.add_argument('--exp-name', default="cifar-simple", help='experiment name')
    parser.add_argument('--subset', default=None, type=int)

    # general
    parser.add_argument('--batch-size', type=int, default=40, metavar='N', help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=260, metavar='N', help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    # parser.add_argument('--gamma', type=float, default=0.1, metavar='M', help='Learning rate step gamma')
    # parser.add_argument('--step-size', type=float, default=40, help='Learning rate step size')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', type=int, default=10, help='save model every k epochs')
    parser.add_argument('--test-model', type=int, default=5, help='test model every k epochs')
    parser.add_argument('--view-loss', type=int, default=400, help='view loss every k epochs')
    parser.add_argument('--plot-interval',  type=int, default=100)
    parser.add_argument('--data_root', default='/tmp/public_dataset/pytorch/', help='folder to save the model')

    # model parameters
    parser.add_argument('--data-set', default='cifar10', help='dataset name')
    parser.add_argument('--model-name', default='simple', help='model name')
    args = parser.parse_args()
    return args



def view_loss(total):
    fig = plt.figure()
    base = range(len(total["loss"]))
    ax0 = fig.add_subplot(1, 1, 1)
    for key in total:
        ax0.plot(base, total[key])

    plt.show()

def test(model):
    test_len = len(test_loader.dataset)
    model.eval()
    acc = 0
    counter = 0
    totaloss = 0

    for batch_idx, (data, target) in enumerate(test_loader):
        with torch.no_grad():
            # print(i)
            # i += 1
            data, target = data.cuda(), target.cuda()
            counter += data.shape[0]
            output = model(data)
            correct = output.data.max(1)[1].eq(target)
            acc += correct.float().sum()
            loss = F.cross_entropy(output, target)
            totaloss += loss.item() * data.size(0)

    test_acc = acc / counter  # average over number of mini-batch
    print('\tTest set: Average loss: {:.4f}'.format(
        totaloss/test_len))
    print('\tTest set: Accuracy: {:.4f}'.format(
        test_acc))
    return test_acc, totaloss/test_len

args = parse_args()



if args.data_set == "cifar10":
    test_loader = dataset.CIFARData.getTestSetIterator(batch_size=args.batch_size)
    train_loader = dataset.CIFARData.getTrainSetIterator(batch_size=args.batch_size)

if args.model_name == "resnet18":
    model = torchvision.models.resnet18()
if args.model_name == "resnet50":
    model = torchvision.models.resnet50()
if args.model_name == "simple":
    model = simple_cifar10(128)
if args.model_name == "resnext":
    model = CifarResNeXt(
        cardinality=8,
        num_classes=10,
        depth=29,
        widen_factor=4,
        dropRate=0,
    )
if args.model_name == "vgg16":
    model = torchvision.models.vgg16_bn(pretrained=False)
    input_lastLayer = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(input_lastLayer, 10)
    print(model.classifier[6])
    # input_lastLayer = model.classifier[6].in_features

utils.mkdir_ifnotexists(utils.concat_home_dir(os.path.join('classifier')))
utils.mkdir_ifnotexists(utils.concat_home_dir(os.path.join('classifier', 'cifar10')))
utils.mkdir_ifnotexists(utils.concat_home_dir(os.path.join('classifier', 'cifar10', args.model_name)))
utils.mkdir_ifnotexists(utils.concat_home_dir(os.path.join('classifier', 'cifar10', args.model_name, args.exp_name)))
utils.mkdir_ifnotexists(
    utils.concat_home_dir(os.path.join('classifier', 'cifar10', args.model_name, args.exp_name, 'checkpoints')))
checkpoint_dir = utils.concat_home_dir(os.path.join('classifier', 'cifar10', args.model_name, args.exp_name, 'checkpoints'))

utils.mkdir_ifnotexists(
    utils.concat_home_dir(os.path.join('classifier', 'cifar10', args.model_name, args.exp_name, 'log')))
logdir = utils.concat_home_dir(os.path.join('classifier', 'cifar10', args.model_name, args.exp_name, 'log'))
logger = utils.Logger()
logger.init(logdir, 'train_log')
print = logger.info

boarddir = utils.concat_home_dir(os.path.join('boards', 'classifier', 'cifar10', args.model_name, args.exp_name))
writer = SummaryWriter(log_dir=boarddir)
print("=================FLAGS==================")
for k, v in args.__dict__.items():
    print('{}: {}'.format(k, v))
    writer.add_text("parameters", '{}: {}'.format(k, v))
print("========================================")


print(checkpoint_dir)
kwargs = {'num_workers': 0, 'pin_memory': True}

model = nn.Sequential(Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), model)
model = nn.DataParallel(model)
params = list(model.parameters())

# optimizer = optim.SGD(params, lr=args.lr, weight_decay=5e-4, momentum=0.9)  # model.parameters()
# optimizer = optim.Adam(params, lr=args.lr)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
# scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
# scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs)
model.cuda()

track = {'loss': []}
start_epoch = 0
dir = [int(i.split('_')[-1][:-3]) for i in os.listdir(checkpoint_dir)]
print(dir)
if len(dir) > 0:
    print(dir)
    start_epoch = sorted(dir)[-1] + 1
    check_pt = torch.load(os.path.join(checkpoint_dir, 'epoch_{}.pt'.format(start_epoch)))
    # args = check_pt['args']
    model.load_state_dict(check_pt['model_state_dict'])
    optimizer.load_state_dict(check_pt['optimizer'])
    # scheduler.load_state_dict(check_pt['scheduler'])


for epoch in range(start_epoch, args.epochs + 1):
    train_len = len(train_loader.dataset)
    model.train()
    totaloss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        # print(data.shape)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        totaloss += loss.item() * data.size(0)

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct = pred.eq(target).sum()
            acc = correct.float() / len(data)
            print('Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f} lr: {:.2e}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                loss.data.item(), acc, optimizer.param_groups[0]['lr']))
            writer.add_scalar('Accuracy/train', acc, epoch)

    track['loss'].append(totaloss/train_len)
    writer.add_scalar('Loss/train', totaloss/train_len, epoch)
    # scheduler.step()

    if (args.save_model > 0) and (epoch % args.save_model == 0):
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'args': args,
                    'optimizer': optimizer.state_dict(),
                    # 'scheduler': scheduler.state_dict(),
                    },
                   os.path.join(checkpoint_dir, 'epoch_{}.pt'.format(str(epoch))))

    if epoch % args.test_model == 0:
        testacc, testloss = test(model)
        writer.add_scalar('Accuracy/test', testacc, epoch)
        writer.add_scalar('Loss/test', testloss, epoch)

    if epoch % args.view_loss == 0 and epoch > 0:
        view_loss(track)

    writer.close()


