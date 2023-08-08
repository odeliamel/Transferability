import argparse
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
# from utils import general as utils
import matplotlib.pyplot as plt
import models.utils as utils
from datasets import dataset
import os

# from managers.cifar_manager import CifarManager

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from torch.utils.tensorboard import SummaryWriter

from threadpoolctl import threadpool_limits

_thread_limit = threadpool_limits(limits=8)


def parse_args():
    parser = argparse.ArgumentParser(description='vgg16 based Encoder-Decoder CIFAR')

    # experiment data
    parser.add_argument('--exp-name', default="cifar-l2", help='experiment name')
    # parser.add_argument('--ds-name', defaulta='COIL20')
    parser.add_argument('--subset', default=None, type=int)
    parser.add_argument('--decoder', default='MyCnnD')
    parser.add_argument('--encoder', default='MyCnnE')
    parser.add_argument('--train-step', default='iae')
    parser.add_argument('--gpu', default=0, help='the GPU number to run on')
    parser.add_argument('--ngpu', default=2, help='the GPU number to run on')


    # general
    parser.add_argument('--batch-size', type=int, default=40, metavar='N', help='input batch size for training')
    parser.add_argument('--sampler', type=str, default='GaussianSampler')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N', help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--gamma', type=float, default=1, metavar='M', help='Learning rate step gamma')
    parser.add_argument('--step-size', type=float, default=1, help='Learning rate step size')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', type=int, default=20, help='save model every k epochs')
    parser.add_argument('--test-model', type=int, default=10, help='view loss every k epochs')

    parser.add_argument('--view-loss', type=int, default=400, help='view loss every k epochs')
    parser.add_argument('--load-model', default=False, help='Load saved model')
    parser.add_argument('--plot-interval',  type=int, default=10000)
    parser.add_argument('--data_root', default='/tmp/public_dataset/pytorch/', help='folder to save the model')

    # model parameters
    parser.add_argument('--nc', default=3, type=int, help='number of channels')
    parser.add_argument('--latent-dim', default=128, type=int, help='latent dimension')
    parser.add_argument('--softplus', default=100, type=float, help='softplus coefficient')
    parser.add_argument('--iso-weight', default=0.075, type=float, help='weight of the isometry loss')
    parser.add_argument('--data-set', default='cifar10', help='dataset name')
    parser.add_argument('--model-name', default='iae', help='model name')
    args = parser.parse_args()
    return args


def view_loss(total):
    fig = plt.figure()
    base = range(len(total["loss"]))
    ax0 = fig.add_subplot(1, 1, 1)
    for key in total:
        ax0.plot(base, total[key])

    plt.show()

def test(model_D, model_E):
    model_E.eval()
    model_D.eval()
    # test_loader_s = save_and_load.load_images_from_folder(folder="cifar10", name="plianncars-val",
    #                                                       batch_size=args.batch_size)
    test_loader_s = dataset.CIFARData.getTestSetIterator(batch_size=args.batch_size)
    test_len = len(test_loader_s.dataset)
    test_loss = 0

    for batch_idx, (data, target) in enumerate(test_loader_s):
        with torch.no_grad():
            # print(i)
            # i += 1
            data, target = data.cuda(), target.cuda()
            output = model_D(model_E(data))
            test_loss += (output - data).pow(2).mean()

        # if batch_idx == 0:
        #     img = data[4].unsqueeze(0)
        #     encdecimg = model_D(model_E(img))
            # print(img.shape, encdecimg.shape)
            # views.basicview.view_classification_changes(CifarManager(), img, encdecimg)

    test_loss = test_loss / test_len  # average over number of mini-batch

    print('\tTest set: Average loss: {:.4f}'.format(
        test_loss))


args = parse_args()

utils.mkdir_ifnotexists(utils.concat_home_dir(os.path.join('autoencoders')))
utils.mkdir_ifnotexists(utils.concat_home_dir(os.path.join('autoencoders', 'cifar10')))
utils.mkdir_ifnotexists(utils.concat_home_dir(os.path.join('autoencoders', 'cifar10', args.model_name)))
utils.mkdir_ifnotexists(utils.concat_home_dir(os.path.join('autoencoders', 'cifar10', args.model_name, args.exp_name)))
utils.mkdir_ifnotexists(
    utils.concat_home_dir(os.path.join('autoencoders', 'cifar10', args.model_name, args.exp_name, 'checkpoints')))
checkpoint_dir = utils.concat_home_dir(os.path.join('autoencoders', 'cifar10', args.model_name, args.exp_name, 'checkpoints'))

utils.mkdir_ifnotexists(
    utils.concat_home_dir(os.path.join('autoencoders', 'cifar10', args.model_name, args.exp_name, 'log')))
logdir = utils.concat_home_dir(os.path.join('autoencoders', 'cifar10', args.model_name, args.exp_name, 'log'))
logger = utils.Logger()
logger.init(logdir, 'train_log')
print = logger.info

boarddir = utils.concat_home_dir(os.path.join('boards', 'autoencoders', 'cifar10', args.model_name, args.exp_name))
writer = SummaryWriter(log_dir=boarddir)
print("=================FLAGS==================")
for k, v in args.__dict__.items():
    print('{}: {}'.format(k, v))
    writer.add_text("parameters", '{}: {}'.format(k, v))
print("========================================")


print(checkpoint_dir)
kwargs = {'num_workers': 0, 'pin_memory': True}


if args.model_name == "vgg16":
    from models.autoencoders.cifar.vgg_model import *

if args.model_name == "simple":
    from models.autoencoders.cifar.simple_model import *

if args.model_name == "iae":
    from models.autoencoders.cifar.iae_model import *

model_E = SegNet_E(args)
model_D = SegNet_D(args)

model_E = nn.Sequential(Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), model_E)
model_D = nn.Sequential(model_D, DeNormalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))

model_E = nn.DataParallel(model_E)
model_D = nn.DataParallel(model_D)

params = list(model_E.parameters()) + list(model_D.parameters())
optimizer = optim.Adam(params, lr=args.lr)  # model.parameters()
scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
# scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs)
model_D.cuda()
model_E.cuda()
track = {'loss': [], 'rec': []}
start_epoch = 0
dir = [int(i.split('_')[-1][:-3]) for i in os.listdir(checkpoint_dir)]
if len(dir) > 0:
    print(dir)
    start_epoch = sorted(dir)[-1]
    check_pt = torch.load(os.path.join(checkpoint_dir, 'epoch_{}.pt'.format(start_epoch)))
    # args = check_pt['args']
    model_E.load_state_dict(check_pt['model_E_state_dict'])
    model_D.load_state_dict(check_pt['model_D_state_dict'])
    optimizer.load_state_dict(check_pt['optimizer'])
    scheduler.load_state_dict(check_pt['scheduler'])


for epoch in range(start_epoch, args.epochs + 1):
    # train_loader = save_and_load.load_images_from_folder(folder="cifar10", name="plianncars",
    #                                                      batch_size=args.batch_size)
    train_loader = dataset.CIFARData.getTrainSetIterator(batch_size=args.batch_size)
    train_len = len(train_loader.dataset)
    # print(train_len)
    model_E.train()
    model_D.train()

    total = {}

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.cuda()
        # print(data.shape)
        optimizer.zero_grad()
        # out = iae_step(args, data, model_E, model_D, sampler, batch_idx, epoch, train_loader)
        out = simple_step(args, data, model_E, model_D, batch_idx, epoch, train_loader, train_len)
        out['loss'].backward()
        optimizer.step()

        for key in out:
            if key in total:
                total[key] += out[key].item() * data.size(0)
            else:
                total[key] = out[key].item() * data.size(0)

    for key in total:
        # print(key)
        total[key] /= len(train_loader)
        track[key].append(total[key])

    # print(track)
    # experiment.log_metrics(total, epoch=epoch)

    scheduler.step()

    if (args.save_model > 0) and (epoch % args.save_model == 0):
        torch.save({'epoch': epoch,
                    'model_E_state_dict': model_E.state_dict(),
                    'model_D_state_dict': model_D.state_dict(),
                    'args': args,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    },
                   os.path.join(checkpoint_dir, 'epoch_{}.pt'.format(str(epoch))))

    if (epoch % args.test_model == 0):
        test(model_D, model_E)

    if (epoch % args.view_loss == 0 and epoch > 0):
        view_loss(track)


