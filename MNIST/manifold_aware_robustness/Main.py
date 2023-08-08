import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from advertorch.attacks import PGDAttack
from algebra.algebraTools import angle_between_tensor
from attacks.PGD import PGDAttack as PGDAttack_nopredict
from attacks.auto_pgd import APGD
from attacks.losses import margin_loss
from datasets import dataset
from views import basicview

from MNIST.manifold_aware_robustness import robustness_test, projections
from MNIST.manifold_aware_robustness.GreedyPPGD import GreedyPPGDAttack
from MNIST.train import get_train_args, train
from MNIST.model import MLP1Classifier, MLPClassifier

from threadpoolctl import threadpool_limits

_thread_limit = threadpool_limits(limits=8)

from highDimSynthetic.sphere.GetParams import get_args
from highDimSynthetic.sphere.CreateData import create_data
from MNIST.manifold_aware_robustness.CreateModel import create_model
from highDimSynthetic.sphere.utils import *
from highDimSynthetic.sphere.AdvPert import *
from MNIST.manifold_aware_robustness.projections import project_diff_on_global_basis, project_diff_off_global_basis

from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler
import torch


def main():
    args = get_args(sys.argv[1:])
    train_args = get_train_args()
    # model1 = eval("MLPClassifier")()
    # trained_model1 = train(train_args, model1)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:{}".format(args.gpu) if args.cuda else "cpu")
    # trained_model1 = trained_model1.eval().cuda()

    # greedy_attack1 = GreedyPPGDAttack(predict=trained_model1, eps=5., nb_iter=400, eps_iter=.05, ord=2,
    #                                          targeted=False,
    #                                          loss_fn=margin_loss,
    #                                          clip_min=-1, clip_max=1, rand_init=False)

    def identity(x, y):
        return x
    all_projections = [identity, project_diff_on_global_basis, project_diff_off_global_basis]
    # factors = list(range(1, 100, 15))
    factors = [0.2, 0.5] + list(range(1, 8, 2))
    # factors = [5]
    # distances = {"identity": [], "project_diff_on_global_basis": [], "project_diff_off_global_basis": []}
    distances = {"identity": torch.zeros([len(factors), 5]), "project_diff_on_global_basis": torch.zeros([len(factors), 5]), "project_diff_off_global_basis": torch.zeros([len(factors), 5])}

    for j in range(5):
        for f in range(len(factors)):
            factor = factors[f]
            print(factor)
            # logdir = 'MNIST/log/robustness/3layers_fc' + "/" + str(factor) + "/"
            if j == 0:
                logdir = '/home/odeliam/PycharmProjects/Transferability/MNIST/log/robustness/3layers_fc' + \
                         "/" + str(factor) + "/"
            else:
                logdir = '/home/odeliam/PycharmProjects/Transferability/MNIST/log/robustness/3layers_fc' + \
                         "/" + str(factor) + "/" + str(j) + "/"
            model = create_model(args)
            if not os.path.exists(logdir):
                adv_trained_model_unit = train(args=train_args, model=model, weight_factor=factor, logdir=logdir)
                adv_trained_model = adv_trained_model_unit.eval().cuda()

            else:
                files = os.listdir(logdir)
                best = [file_name for file_name in files if "best" in file_name]
                checkpt = torch.load(os.path.join(logdir, best[0]))
                model.load_state_dict(checkpt)
                adv_trained_model = model.eval().cuda()

                greedy_attack = GreedyPPGDAttack(predict=adv_trained_model, eps=60., nb_iter=8000, eps_iter=.05, ord=2,
                                                 targeted=False,
                                                 loss_fn=margin_loss,
                                                 clip_min=-1, clip_max=1, rand_init=False)
                d = robustness_test.test_robustness(greedy_attack, all_projections)
                for key in distances.keys():
                    distances[key][f, j] = torch.mean(d[key]).item()
        # return
    print(distances)
    import matplotlib.pyplot as plt
    import numpy as np
    max_distances = {"identity": [], "project_diff_on_global_basis": [], "project_diff_off_global_basis": []}
    min_distances = {"identity": [], "project_diff_on_global_basis": [], "project_diff_off_global_basis": []}
    first_distances = {"identity": [], "project_diff_on_global_basis": [], "project_diff_off_global_basis": []}
    for key in distances.keys():
        for i in range(len(factors)):
            print("2", distances[key][i])
            tmax = max(distances[key][i])
            tmin = min(distances[key][i])
            tmean = torch.mean(distances[key][i])
            first_distances[key].append(tmean)
            max_distances[key].append(tmax - tmean)
            min_distances[key].append(tmean - tmin)
    print(max_distances)
    for key in distances.keys():
        # plt.plot(factors, distances[key], label=key)
        # plt.plot(factors, distances[key], label=key)
        print(first_distances[key])
        plt.errorbar(factors, first_distances[key], yerr=[min_distances[key], max_distances[key]], label=key,
                     ecolor='black', capsize=2)
    plt.legend()
    plt.xticks([0.5] + list(np.arange(1, 8, step=1)))
    plt.xlim((0.4, 7.1))
    plt.show()


if __name__ == '__main__':
    main()
