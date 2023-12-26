import torch
import os

from attacks.auto_pgd import APGD

from MNIST.robustness.cross_robustness_test import cross_test_w_dist

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from classifiers.abstract_classifier import Classifier
from datasets.dataset import CIFAR10, IMAGENET, MNIST, CIFAR100, RobustbenchIMAGENET

from attacks.PGD import PGDAttack
from attacks.losses import margin_loss
from attacks.abstract_attack import RegAttackWrapper

from MNIST.model import MLPClassifier, MLPnPCAClassifier, MLP1Classifier


def generate_benchmark(attack_tester, models):
    for attack_test in attack_tester:
        for model in models:
            acc = model.get_accuracy()
            print(model.model_name, acc)
            asr = attack_test.test()(model)
            asr_b = asr - (1-acc)
            print('asr_bruto: {:.2%}'.format(asr_b))
            # print(model.model_name, asr)


def generate_cross_benchmark(attack_tester, models, sur_model):
    for attack_test in attack_tester:
        for model in models:
            acc = model.get_accuracy()
            print(model.model_name, acc)
            asr = cross_test_w_dist(model, attack_test, sur_model)
            # asr = attack_test.test()(model)
            asr_b = asr - (1-acc)
            print('asr_bruto: {:.2%}'.format(asr_b))
            # print(model.model_name, asr)



def main():
    attacks = [
        # RegAttackWrapper(PGDAttack(eps=1., nb_iter=50, eps_iter=2*1./50,
        #                    rand_init=False, ord=2, targeted=False, clip_min=0., clip_max=1.,
        #                    loss_fn=margin_loss)),
        RegAttackWrapper(APGD(eps=1., norm=2, targeted=False)),
    ]

    model1 = MLP1Classifier()
    model1.load_state_dict(
        torch.load("/home/odeliam/PycharmProjects/Transferability/MNIST/log/fc1_new/mnist_clean1200/latest.pth"))
    model1 = Classifier(model_name="clean", model=model1, dataset=MNIST)

    model2 = MLP1Classifier()
    model2.load_state_dict(
        torch.load("/home/odeliam/PycharmProjects/Transferability/MNIST/log/fc1_new/classic_advtrain1200/latest.pth"))
    model2 = Classifier(model_name="classic adv train", model=model2, dataset=MNIST)


    models = [
        model1,
        model2,
    ]

    global_dims = [100]
    logs = ['log/fc1_new/projected/shafi_advtrain_testand1train_projected',
            'log/fc1_new/projected/shafi_advtrain_testand1train_projected_pca100q_',
            'log/fc1_new/projected/shafi_advtrain_testand1train_projected_pca32q_',
            'log/fc1_new/projected/shafi_advtrain_testand1train_projected_pca32q_sameinit_']
    for d in global_dims:
        for log in logs:
            m = MLP1Classifier()
            m.load_state_dict(torch.load(log + str(d) + "/latest.pth"))
            m = Classifier(model_name="shafi adv train dim "+str(d), model=m, dataset=MNIST)
            models.append(m)

    generate_benchmark(attacks, models)
    # generate_cross_benchmark(attacks, models, models[0])

if __name__ == '__main__':
    main()