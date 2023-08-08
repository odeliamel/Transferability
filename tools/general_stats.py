import os

from attacks.EBT import getTarget

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from attacks import PGD
from attacks.losses import margin_loss
import numpy as np
from classifiers.abstract_classifier import load, Classifier
from classifiers.abstract_classifier import load_all_from_names
from models.local_classifiers.cifar import cifar_manager
from datasets.dataset import CIFAR10
from tools.basics import check_transferability
from defences.robustbench_dict import l2_cifar10


def cifar_pretrained_transferability(attack):
    classifiers = [
        # lambda x=1: load(model_name='simple', dataset='cifar10'),
        # lambda x=1: load(model_name='resnet18', dataset='cifar10'),
        # lambda x=1: load(model_name='resnet50', dataset='cifar10'),
        # lambda x=1: load(model_name='densenet', dataset='cifar10'),
        # lambda x=1: load(model_name='vgg16', dataset='cifar10'),
        # lambda x=1: load(model_name='resnext', dataset='cifar10'),
        load(model_name='simple', dataset='cifar10'),
        load(model_name='resnet18', dataset='cifar10'),
        load(model_name='resnet50', dataset='cifar10'),
        load(model_name='densenet', dataset='cifar10'),
        load(model_name='vgg16', dataset='cifar10'),
        load(model_name='resnext', dataset='cifar10'),
        # Classifier(model_name="simple",
        #            model=cifar_manager.load_model(model_name="simple", exp_name="cifar-simple").model,
        #            dataset=CIFAR10),
    ]

    cifar_transferability(classifiers, classifiers, attack1=attack)

def cifar_to_robust_transferability(attack):
    classifiersfrom = [
        # load(model_name='simple', dataset='cifar10'),
        # load(model_name='resnet50', dataset='cifar10'),
        # load(model_name='vgg16', dataset='cifar10'),
    # lambda n='Standard': load(model_name='Standard', threat_model='L2', dataset='cifar10', local=False),
    # lambda n='ResNet': load(model_name='resnet18', dataset='cifar10'),
    lambda n='Sehwag2021Proxy_R18': load(model_name='Sehwag2021Proxy_R18', threat_model='L2', dataset='cifar10', local=False),



    ]
    classifiersto = [
    # lambda n='Sehwag2021Proxy_R18': load(model_name='Sehwag2021Proxy_R18', threat_model='L2', dataset='cifar10', local=False),
    lambda n='ResNet': load(model_name='resnet18', dataset='cifar10')

    ]

    # classifiersto = load_all_from_names(l2_cifar10)
    # for c in [i() for i in classifiersto]:
    #     cifar_transferability([c], [c])
    cifar_transferability([i() for i in classifiersfrom], [i() for i in classifiersto], attack)


def cifar_local_trained_transferability():
    classifiers = [
        Classifier(model_name="vgg16-1",
                   model=cifar_manager.load_model(model_name="vgg16", exp_name="cifar-sgd-new-nopretrain260-2").model,
                   dataset=CIFAR10),
        Classifier(model_name="vgg16-2",
                   model=cifar_manager.load_model(model_name="vgg16", exp_name="cifar-sgd-new-nopretrain260-3").model,
                   dataset=CIFAR10),
        Classifier(model_name="simple",
                   model=cifar_manager.load_model(model_name="simple", exp_name="cifar-simple").model,
                   dataset=CIFAR10),
    ]

    cifar_transferability(classifiers)


def cifar_transferability(classifiersfrom, classifiersto, attack1):
    res_string_from = ""
    res_dict_to = {}
    for classifier1 in classifiersfrom:
        res_string_from += classifier1.model_name
        res_string_from += " & "
        iters = 50
        # eps = 16/255
        # eps = 0.5
        eps = 1.0
        eps_iter = eps / iters
        loss_fn = margin_loss
        try:
            test_set = classifier1.dataset.getTestSetIterator(batch_size=100)

            # find_target = lambda data, target: target
            find_target = lambda data, target: getTarget.get_best_target(classifier1, data, target, epsilon=eps, ord=2,
                                                                         loss_fn=loss_fn)

            for classifier2 in classifiersto:
                if classifier2.model_name not in res_dict_to.keys():
                    res_dict_to[classifier2.model_name] = ""
                else:
                    res_dict_to[classifier2.model_name] += " & "

                asr, same_class_transfer, any_class_transfer = check_transferability(model1=classifier1, model2=classifier2,
                                                                                     attack=attack1, test_loader=test_set,
                                                                                     target_fn=find_target)

                print("model 1: {} (acc: {}), model 2: {} (acc:{}), \n"
                      "asr: {}, same class transferability: {}, any class transferability: {}".format(
                        classifier1.model_name, classifier1.get_accuracy(), classifier2.model_name, classifier2.get_accuracy(),
                        asr, same_class_transfer, any_class_transfer
                        ))
                res_dict_to[classifier2.model_name] += "asr: {:.2%}, transfer: {:.2%} sameclass: {:.2%}".format(
                                                            asr, any_class_transfer, same_class_transfer)
                res_dict_to[classifier2.model_name] += " & "
        except Exception as e:
            print(type(e))  # the exception instance
            print(e.args)  # arguments stored in .args
            print(e) # __str__ allows args to be printed directly,
            continue

    print(res_string_from)
    print(res_dict_to)




if __name__ == '__main__':
    iters = 50
    loss_fn = margin_loss
    # cifar_local_trained_transferability()
    attack1 = PGD.PGDAttack(eps=1.0, nb_iter=iters, eps_iter=6.0/50,
                            rand_init=True, ord=2, targeted=True, loss_fn=loss_fn)
    attack2 = PGD.PGDAttack(eps=16/255, nb_iter=iters, eps_iter=2*16/255/50,
                            rand_init=False, ord=np.inf, targeted=True, loss_fn=loss_fn)
    # cifar_pretrained_transferability(attack1)
    # cifar_pretrained_transferability(attack2)

    cifar_to_robust_transferability(attack1)
