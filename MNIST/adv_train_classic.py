import argparse
import os
import time
import os

from advertorch.attacks import PGDAttack
from algebra.projectionTools import get_projection_matrix

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from Autoencoders.mnist_encoder.autoencoder import MnistEncoder
from algebra import projectionTools
from algebra.algebraTools import create_orthonormal_basis
from attacks.PPGD import PPGDAttack
from attacks.losses import margin_loss

from MNIST.PCAMNIST import load_PCA

# import trained.mnist.dataset as dataset
from datasets import dataset
# import trained.mnist.model as model
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
# from trained import misc
from classifiers import utils
# from managers.mnist_encoder import MnistEncoder
from MNIST.model import MLPClassifier, MLPnPCAClassifier, MLP1Classifier
from threadpoolctl import threadpool_limits
_thread_limit = threadpool_limits(limits=8)


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--batch_size', type=int, default=200, help='input batch size for training (defaulta: 64)')
parser.add_argument('--epochs', type=int, default=12000, help='number of epochs to train (defaulta: 10)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (defaulta: 1e-3)')
parser.add_argument('--gpu', default=None, help='index of gpus to use')
parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to use')
parser.add_argument('--seed', type=int, default=117, help='random seed (defaulta: 1)')
parser.add_argument('--log_interval', type=int, default=100,  help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=5,  help='how many epochs to wait before another test')
parser.add_argument('--logdir', default='log/fc1_new/classic_advtrain1200', help='folder to save to the log')
parser.add_argument('--data_root', default='/tmp/public_dataset/pytorch/', help='folder to save the model')
parser.add_argument('--decreasing_lr', default='40,80', help='decreasing strategy')
parser.add_argument('--weight_factor', type=int, default=1, help='modelclass')

# parser.add_argument('--model', default='MLPnPCAClassifier', help='modelclass')
parser.add_argument('--model', default='MLP1Classifier', help='modelclass')
# parser.add_argument('--low_dim', type=int, default=784, help='the dimension reductions destination dimension')

# parser.add_argument('--global_manifold_dim', type=int, default=30, help='the PCA linear manifold dimension')


def train(model=None, train_loader=None, test_loader=None):
    args = parser.parse_args()
    args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
    logger = utils.Logger()
    logger.init(args.logdir, 'train_log')
    print = logger.info

    # select gpu
    # args.gpu = misc.auto_select_gpu(utility_bound=0, num_gpu=args.ngpu, selected_gpus=args.gpu)
    # args.ngpu = len(args.gpu)

    # logger
    # misc.ensure_dir(args.logdir)
    print("=================FLAGS==================")
    for k, v in args.__dict__.items():
        print('{}: {}'.format(k, v))
    print("========================================")

    # seed
    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # data loader
    # train_loader, test_loader = dataset.get(batch_size=args.batch_size, data_root=args.data_root, num_workers=1)
    if train_loader is None:
        train_loader = dataset.MNIST.getTrainSetIterator(batch_size=args.batch_size)
    if test_loader is None:
        test_loader = dataset.MNIST.getTestSetIterator(batch_size=args.batch_size)

    # model
    # model = model.mnist(input_dims=784, n_hiddens=[256, 256], n_class=10)
    if model is None:
        model = MLPClassifier()
        # model = model.mnist(input_dims=784, n_hiddens=[128, 128], n_class=10)
    # factor = args.factor
    model.fc1.weight = torch.nn.Parameter(model.fc1.weight / args.weight_factor)
    model.fc1.bias = torch.nn.Parameter(model.fc1.bias / args.weight_factor)

    if args.cuda:
        model.cuda()

    model = torch.nn.DataParallel(model)

    # optimizer
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
    print('decreasing_lr: ' + str(decreasing_lr))
    utils.model_snapshot(model, os.path.join(args.logdir, 'init.pth'))

    encoder = MnistEncoder()
    attack = PGDAttack(predict=model, eps=1., nb_iter=1, eps_iter=1.,
                           rand_init=False, ord=2, targeted=False, clip_min=0., clip_max=1.,
                           loss_fn=margin_loss)

    best_acc, old_file = 0, None
    t_begin = time.time()
    try:
        # ready to go
        for epoch in range(args.epochs):
            model.train()
            if epoch in decreasing_lr:
                optimizer.param_groups[0]['lr'] *= 0.1
            for batch_idx, (data, target) in enumerate(train_loader):
                indx_target = target.clone()
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                # a = attack.perturb(data, target)
                run_adv_epoch(attack, args, batch_idx, data, epoch, F.cross_entropy,
                              model, optimizer, target, len(train_loader.dataset), print)

                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

            elapse_time = time.time() - t_begin
            speed_epoch = elapse_time / (epoch + 1)
            speed_batch = speed_epoch / len(train_loader)
            eta = speed_epoch * args.epochs - elapse_time
            print("Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
                elapse_time, speed_epoch, speed_batch, eta))
            utils.model_snapshot(model, os.path.join(args.logdir, 'latest.pth'))

            if epoch % args.test_interval == 0:
                model.eval()
                test_loss = 0
                correct = 0
                for data, target in test_loader:
                    indx_target = target.clone()
                    if args.cuda:
                        data, target = data.cuda(), target.cuda()
                    data, target = Variable(data, volatile=True), Variable(target)
                    output = model(data)
                    test_loss += F.cross_entropy(output, target).data.item()
                    pred = output.data.max(1)[1]  # get the index of the max log-probability
                    correct += pred.cpu().eq(indx_target).sum()

                test_loss = test_loss / len(test_loader) # average over number of mini-batch
                acc = 100. * correct / len(test_loader.dataset)
                print('\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                    test_loss, correct, len(test_loader.dataset), acc))
                if acc > best_acc:
                    new_file = os.path.join(args.logdir, 'best-{}.pth'.format(epoch))
                    utils.model_snapshot(model, new_file, old_file=old_file, verbose=True)
                    best_acc = acc
                    old_file = new_file
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        print("Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time()-t_begin, best_acc))


def run_adv_epoch(attack, args, batch_idx, data, epoch, loss_fn, model, optimizer, target, train_len, print):
    # print("step")
    optimizer.zero_grad()
    model.eval()

    adv = data + attack.perturb(data, target)

    model.zero_grad()
    model.train()
    output = model(adv)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()

    if batch_idx % args.log_interval == 0:  # and batch_idx > 0:
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct = pred.eq(target).sum()
        acc = correct * 1.0 / len(data)
        print('Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f} lr: {:.2e}'.format(
            epoch, batch_idx * len(data), train_len,
            loss.data.item(), acc, optimizer.param_groups[0]['lr']))



if __name__ == '__main__':
    args = parser.parse_args()
    model = eval(args.model)()#(num_of_dims=args.low_dim)
    train(model)

