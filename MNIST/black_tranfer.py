import torch
import torch.nn as nn
import os

from algebra.algebraTools import project_on_basis

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from MNIST.model import MLPnPCAClassifier, MLPClassifier

from datasets import dataset
import advertorch.attacks.iterative_projected_gradient as PGD
from attacks.losses import imagenet_margin_loss, margin_loss
import sys
sys.path.insert(1, '/home/odeliam/PycharmProjects/EBT')
import attacks.EBT.getTarget as getTarget
from algebra import projectionTools
import algebra.algebraTools as algebraTools
from views import basicview
from tools.basics import check_transferability
# from algebra.on_manifold_attack import OnManifoldPGDAttack
from Autoencoders.mnist_encoder.autoencoder import MnistEncoder
from classifiers.MNIST import model


def exp1():
    model1 = MLPClassifier()
    model2 = MLPClassifier()

    model1.load_state_dict(torch.load("/home/odeliam/PycharmProjects/Transferability/MNIST/log/non-robust-fc/best-30.pth"))
    model2.load_state_dict(torch.load("/home/odeliam/PycharmProjects/Transferability/MNIST/log/non-robust-fc2/best-25.pth"))

    model1, model2 = model1.eval(), model2.eval()
    model1, model2 = model1.cuda(), model2.cuda()

    # encoder = MnistEncoder()

    on_manifold_angles = torch.tensor([]).cuda()
    off_manifold_angles = torch.tensor([]).cuda()

    # eps = 16/255
    # eps_iter = 2 * eps / 50
    eps = 1.0
    eps_iter = eps
    loss_fn = margin_loss

    test_set = dataset.MNIST.getTestSetIterator(batch_size=1)

    attack1 = PGD.PGDAttack(predict=model1, eps=eps, nb_iter=1, eps_iter=eps_iter,
                           rand_init=False, ord=2, targeted=False, clip_min=0., clip_max=1.,
                           loss_fn=loss_fn)

    attack2 = PGD.PGDAttack(predict=model2, eps=eps, nb_iter=1, eps_iter=eps_iter,
                           rand_init=False, ord=2, targeted=False, clip_min=0., clip_max=1.,
                           loss_fn=loss_fn)

    for data, target in test_set:
        print("------------------------------")
        data, target = data.cuda(), target.cuda()
        # adv_target1 = getTarget.get_best_target(model1.model, data, target, epsilon=eps, ord=2,
        #                                        loss_fn=loss_fn)
        # adv_target2 = getTarget.get_best_target(model2.model, data, target, epsilon=eps, ord=2,
        #                                         loss_fn=loss_fn)
        adv_target1 = target
        adv_target2 = target

        print("original target: ", target.item())
        model1_classification = model1(data).data.max(1)[1]
        model2_classification = model2(data).data.max(1)[1]
        print("current classification by 1: {}, by 2: {}".format(model1_classification, model2_classification))
        print("suggested best targets: 1 - {}, 2 - {}".format(adv_target1.item(), adv_target2.item()))

        if adv_target1.item() != adv_target2.item() or model1_classification.item() != model2_classification.item()\
                or model1_classification.item() != target.item() or model2_classification.item() != target.item():
            continue

        adv_m1 = attack1.perturb(data, adv_target1)
        adv_m2 = attack2.perturb(data, adv_target2)

        print("angle between adversarial vectors : ", algebraTools.angle_between_tensor(adv_m1 - data, adv_m2 - data))
        # print("angle between adversarial vectors - the diff norm : ", torch.norm((adv_m1 - adv_m2), p=2))

        # encoded_image = encoder.encode_and_decode(data)

        adv1_diff = adv_m1 - data
        adv2_diff = adv_m2 - data
        #
        # print("AE diff norm: ", torch.norm((encoded_image - data), p=2))
        # print("angle between adversarial vectors - the diff from adv to AE image", algebraTools.angle_between_tensor(adv1_diff_to_ae, adv2_diff_to_ae))

        # basis = projectionTools.find_local_manifold_around_image(encoder, data)

        # data_proj = encoded_image + projectionTools.project_diff_on_basis(basis, data - encoded_image)

        # print("projection diff norm:", torch.norm(data_proj - data))
        # adv1_proj = encoded_image + projectionTools.project_diff_on_basis(basis, adv1_diff_to_ae)
        # adv2_proj = encoded_image + projectionTools.project_diff_on_basis(basis, adv2_diff_to_ae)

        def project_diff_on_basis(x):
            res = x.clone()
            # print(res.shape)
            res[:, :, :, 0] = 0
            # res[:, :, :, 27] = 0
            # res[:, :, 0, :] = 0
            # res[:, :, 27, :] = 0
            return res

        adv_m1_proj_on = data + project_diff_on_basis(adv1_diff)
        adv_m2_proj_on = data + project_diff_on_basis(adv2_diff)

        print("on manifold distances:")
        print("on manifold adversarial distance 1:", torch.norm(adv_m1_proj_on - data, p=2))
        print("on manifold adversarial distance 2:", torch.norm(adv_m2_proj_on - data, p=2))
        on_manifold_angle = algebraTools.angle_between_tensor(adv_m1_proj_on - data, adv_m2_proj_on - data)
        print("on manifold angle between adversarial vectors:", on_manifold_angle)
        on_manifold_angles = torch.cat((on_manifold_angles, on_manifold_angle.unsqueeze(0)), dim=0)

        # data_off_manifold = data - data_proj
        adv_m1_off_manifold = data + (adv1_diff - project_diff_on_basis(adv1_diff))
        adv_m2_off_manifold = data + (adv2_diff - project_diff_on_basis(adv2_diff))

        print("off manifold adversarial distance 1:", torch.norm(adv_m1_off_manifold - data, p=2))
        print("off manifold adversarial distance 2:", torch.norm(adv_m2_off_manifold - data, p=2))
        off_manifold_angle = algebraTools.angle_between_tensor(adv_m1_off_manifold - data, adv_m2_off_manifold - data)
        print("off manifold angle between adversarial vectors:", off_manifold_angle)
        off_manifold_angles = torch.cat((off_manifold_angles, off_manifold_angle.unsqueeze(0)), dim=0)

        #TODO: see if the margin between classes is closing like linear function with cosine ratio!

        # basicview.view_classification_changes_multirow([model1]*4, [data]*4, [adv_m1, adv_m1_off_manifold, adv_m2, adv_m2_off_manifold],
        #                                                txt=["model 1, adv 1", "model 1 adv 1 proj off","model 1 adv 2", "model 1 adv2 proj off"])
        # basicview.view_classification_changes_multirow([model2]*4, [data]*4, [adv_m1, adv_m1_off_manifold, adv_m2, adv_m2_off_manifold],
        #                                                txt=["model 2, adv 1", "model 2 adv 1 proj off","model 2 adv 2", "model 2 adv2 proj off"])

    print("on manifold angle mean:", on_manifold_angles.mean())
    print("off manifold angle mean: ", off_manifold_angles.mean())



if __name__ == '__main__':
    # exp0()
    exp1()
    # exp2()
    # exp3()