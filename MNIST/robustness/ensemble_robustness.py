import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from classifiers.abstract_classifier import Classifier
from datasets.dataset import MNIST

import argparse
from MNIST.model import MLP1Classifier
import torch
from attacks.abstract_attack import RegAttackWrapper
from attacks.auto_pgd import APGD

from MNIST.robustness.robustness_test import generate_cross_benchmark
from MNIST.train import train

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--batch_size', type=int, default=200, help='input batch size for training (defaulta: 64)')
parser.add_argument('--epochs', type=int, default=120, help='number of epochs to train (defaulta: 10)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (defaulta: 1e-3)')
parser.add_argument('--gpu', default=None, help='index of gpus to use')
parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to use')
parser.add_argument('--seed', type=int, default=117, help='random seed (defaulta: 1)')
parser.add_argument('--log_interval', type=int, default=100,  help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=5,  help='how many epochs to wait before another test')
# parser.add_argument('--logdir', default='log/fc1_new/mnist_clean1200', help='folder to save to the log')
parser.add_argument('--data_root', default='/tmp/public_dataset/pytorch/', help='folder to save the model')
parser.add_argument('--decreasing_lr', default='10,25', help='decreasing strategy')

# parser.add_argument('--model', default='MLPnPCAClassifier', help='modelclass')
parser.add_argument('--model', default='MLP1Classifier', help='modelclass')
# parser.add_argument('--low_dim', type=int, default=784, help='the dimension reductions destination dimension')




def main(to_train=False):
    log = 'log/ensemble/mnist_clean120_'
    models = []

    for i in range(2):
        args = parser.parse_args()
        args.logdir = log + str(i)
        model = eval(args.model)()  # (num_of_dims=args.low_dim)
        if to_train:
            args.weight_factor = 1
            train(args, model)
        else:
            model.load_state_dict(
                torch.load(os.path.join("/home/odeliam/PycharmProjects/Transferability/MNIST", log + str(i), "latest.pth")))

        model = Classifier(model_name="clean"+str(i), model=model, dataset=MNIST)
        models.append(model)

    def ensemble_classification(models, x):
        res = torch.zeros(x.shape[0], len(models))
        for k in range(len(models)):
            out = models[k](x)
            res[:, k] = out.data.max(1)[1]
        # return majority vote
        majority_vote, _ = torch.mode(res, 1)
        # print(res[0, :], majority_vote[0])
        return majority_vote.cuda()

    sur_model = lambda x: ensemble_classification(models, x)

    attacks = [
        RegAttackWrapper(APGD(eps=1., norm=2, targeted=False)),
        # RegAttackWrapper(APGD(eps=1., norm=2, targeted=False)),

    ]
    generate_cross_benchmark(attacks, [models[0]], sur_model)
    generate_cross_benchmark(attacks, [models[0]], lambda x: models[0](x).data.max(1)[1])

    model2 = MLP1Classifier()
    model2.load_state_dict(
        torch.load("/home/odeliam/PycharmProjects/Transferability/MNIST/log/fc1_new/classic_advtrain1200/latest.pth"))
    model2 = Classifier(model_name="classic adv train", model=model2, dataset=MNIST)
    generate_cross_benchmark(attacks, [model2], lambda x: model2(x).data.max(1)[1])


if __name__ == '__main__':
    main(to_train=True)