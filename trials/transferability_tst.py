import torch
from algebra import projectionTools
import importlib
import views.basicview as basicview
import os, hashlib


def check_on_manifold_transferability(encoder, model1, model2, attack, test_loader, target_fn):
    fish_basis_home = "/net/mraid11/export/data/odeliam/datasets/imagenet/goldfish-basis"

    attack.predict = model1.model
    correctly_classified_counter = 0
    attack_success = 0
    target_success_counter = 0
    error_counter = 0

    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        autoencoded_image = encoder.encode_and_decode(data)

        model1_classification = model1.model(autoencoded_image).data.max(1)[1]
        model2_classification = model2.model(autoencoded_image).data.max(1)[1]

        correcly_classified = model1_classification.eq(target) & model2_classification.eq(target)
        correctly_classified_counter += correcly_classified.float().sum()
        if correcly_classified.float().sum() == 0:
            continue

        h = hashlib.sha256(data.cpu().numpy()).hexdigest()
        path = fish_basis_home + "/{}.pt".format(str(h))
        if os.path.exists(path):
            print("load basis")
            basis = torch.load(path)
        else:
            basis = projectionTools.find_local_manifold_around_image(encoder, data)

        c_data, c_target = autoencoded_image[correcly_classified], target[correcly_classified]

        adv_target = target_fn(c_data, c_target)

        adv = attack.perturb(c_data, adv_target, basis=basis).detach()

        org_adv_classification = model1.model(adv).data.max(1)[1]
        attack_success += (~org_adv_classification.eq(c_target)).float().sum()

        adv_classification = model2.model(adv).data.max(1)[1]

        target_success_counter += adv_classification.eq(adv_target).float().sum()
        error_counter += (~adv_classification.eq(c_target)).float().sum()

        print("attack succes on original model {}, ({})".format(attack_success, 100 * attack_success / correctly_classified_counter))
        print("target_success_counter {}, ({})".format(target_success_counter, 100 * target_success_counter / correctly_classified_counter))
        print("error_counter: {}, ({})".format(error_counter, 100 * error_counter / correctly_classified_counter))
        print("correctly_classified_counter", correctly_classified_counter)


def check_transferability_retry(model1, model2, attack, test_loader, target_fn, retries=10):
    print("check_transferability")
    attack.predict = model1.model
    correctly_classified_counter = 0
    attack_success = 0
    target_success_counter = 0
    error_counter = 0

    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()

        model1_classification = model1.model(data).data.max(1)[1]
        model2_classification = model2.model(data).data.max(1)[1]

        correcly_classified = model1_classification.eq(target) & model2_classification.eq(target)
        correctly_classified_counter += correcly_classified.float().sum()
        if correcly_classified.float().sum() == 0:
            continue

        to_improve = correcly_classified.clone()
        c_data, c_target = data[to_improve], target[to_improve]

        l_correctly_classified_counter = 0
        l_attack_success = 0
        l_target_success_counter = 0
        l_error_counter = 0
        for i in range(retries):
            for adv_target in target_fn(c_data, c_target):

                # adv_target = target_fn(c_data, c_target)

                adv = attack.perturb(c_data, adv_target).detach()
                adv_classification = model2.model(adv).data.max(1)[1]

                l_error_counter = (~adv_classification.eq(c_target)).float().sum()
                # if l_error_counter == c_data.shape[0]:

        org_adv_classification = model1.model(adv).data.max(1)[1]
        attack_success += (~org_adv_classification.eq(c_target)).float().sum()

        adv_classification = model2.model(adv).data.max(1)[1]

        target_success_counter += adv_classification.eq(adv_target).float().sum()
        error_counter += (~adv_classification.eq(c_target)).float().sum()

        print("attack success on original model {}, ({})".format(attack_success, 100 * attack_success / correctly_classified_counter))
        print("target_success_counter {}, ({})".format(target_success_counter, 100 * target_success_counter / correctly_classified_counter))
        print("error_counter: {}, ({})".format(error_counter, 100 * error_counter / correctly_classified_counter))
        print("correctly_classified_counter", correctly_classified_counter)