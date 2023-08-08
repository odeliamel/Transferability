import torch
from algebra import projectionTools
import importlib
import views.basicview as basicview
import os, hashlib

from attacks.abstract_attack import AbstractAttack



def check_transferability(model1, model2, attack: AbstractAttack, test_loader, target_fn):
    # print("check_transferability")
    # attack.predict = model1.model
    correctly_classified_counter = 0
    attack_success = 0
    target_success_counter = 0
    error_counter = 0
    total_counter = 0

    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        total_counter += data.shape[0]
        model1_classification = model1(data).data.max(1)[1]
        model2_classification = model2(data).data.max(1)[1]

        correcly_classified = model1_classification.eq(target) & model2_classification.eq(target)
        correctly_classified_counter += correcly_classified.float().sum()
        if correcly_classified.float().sum() == 0:
            continue

        c_data, c_target = data[correcly_classified], target[correcly_classified]

        adv_target = target_fn(c_data, c_target)

        adv = attack.attack(model1, c_data, adv_target).detach()

        org_adv_classification = model1(adv).data.max(1)[1]
        attack_success += (~org_adv_classification.eq(c_target)).float().sum()

        adv_classification = model2(adv).data.max(1)[1]

        target_success_counter += (~adv_classification.eq(c_target) & adv_classification.eq(org_adv_classification)).float().sum()
        error_counter += (~adv_classification.eq(c_target)).float().sum()

    # print("total images counter: {}".format(total_counter))
    # print("correctly_classified_counter", correctly_classified_counter)
    # misclassified = total_counter - correctly_classified_counter
    # asr = (attack_success + misclassified) / total_counter
    # same_class_transfer = (target_success_counter + misclassified) / total_counter
    # any_class_transfer = (error_counter + misclassified) / total_counter
    asr =  attack_success / correctly_classified_counter
    # print("attack success on original model {}, ({} from correctly classified)".format(
    #     attack_success, 100 * asr))
    same_class_transfer = target_success_counter / correctly_classified_counter
    # print("Second model was mistaken for the same class as original: {}, ({} from correctly classified)".format(
    #     target_success_counter, 100 * same_class_transfer))
    any_class_transfer = error_counter / correctly_classified_counter
    # print("Second model was mistaken: {}, ({} from correctly classified)".format(
    #     error_counter, 100 * any_class_transfer))


    # print("asr {}".format(asr))
    # print("same class transferability".format(same_class_transfer))
    # print("any class transferability".format(any_class_transfer))
    return asr, same_class_transfer, any_class_transfer
        # basicview.view_classification_changes_multirow(model1, [data], [adv])
        # basicview.view_classification_changes_multirow(model2, [data], [adv])


def check_transferability_with_adv(model1, model2, test_loader, adv, adv_target):
    print("check_transferability")
    correctly_classified_counter = 0
    attack_success = 0
    target_success_counter = 0
    error_counter = 0
    total_counter = 0

    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        total_counter += data.shape[0]
        model1_classification = model1(data).data.max(1)[1]
        model2_classification = model2(data).data.max(1)[1]

        correcly_classified = model1_classification.eq(target) & model2_classification.eq(target)
        correctly_classified_counter += correcly_classified.float().sum()
        if correcly_classified.float().sum() == 0:
            continue

        c_data, c_target = data[correcly_classified], target[correcly_classified]

        adv_target = adv_target[correcly_classified]

        adv = adv[correcly_classified]

        org_adv_classification = model1(adv).data.max(1)[1]
        attack_success += (~org_adv_classification.eq(c_target)).float().sum()

        adv_classification = model2(adv).data.max(1)[1]

        target_success_counter += (~adv_classification.eq(c_target) & adv_classification.eq(org_adv_classification)).float().sum()
        error_counter += (~adv_classification.eq(c_target)).float().sum()

    print("total images counter: {}".format(total_counter))
    print("correctly_classified_counter", correctly_classified_counter)

    asr = 100 * attack_success / correctly_classified_counter
    print("attack success on original model {}, ({} from correctly classified)".format(
        attack_success, asr))
    same_class_transfer = 100 * target_success_counter / correctly_classified_counter
    print("Second model was mistaken for the same class as original: {}, ({} from correctly classified)".format(
        target_success_counter, same_class_transfer))
    any_class_transfer = 100 * error_counter / correctly_classified_counter
    print("Second model was mistaken: {}, ({} from correctly classified)".format(
        error_counter, any_class_transfer))

    return asr, same_class_transfer, any_class_transfer