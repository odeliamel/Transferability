import torch
import torch.nn as nn
import os

from algebra.algebraTools import project_on_basis, create_orthonormal_basis
from attacks.auto_pgd import APGD

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from MNIST.model import MLPnPCAClassifier, MLPClassifier, MLP1Classifier

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



def exp12():
    model1 = MLP1Classifier()
    model2 = MLP1Classifier()
    model3 = MLPnPCAClassifier(num_of_dims=50).cuda()

    def project_diff_on_global_basis(x):
        basis1 = model3.pca.V[:, :model3.pca.k].T
        print(basis1.shape)
        basis1 = create_orthonormal_basis(basis1)
        return projectionTools.project_diff_on_basis(basis1, x)

    def project_diff_off_global_basis(x):
        global_basis = torch.eye(784).cuda()
        global_basis[:model3.pca.k, :] = model3.pca.V.T[:model3.pca.k, :]
        global_basis = create_orthonormal_basis(global_basis)
        global_basis[:model3.pca.k, :] = 0
        return projectionTools.project_diff_on_basis(global_basis, x)

    model1.load_state_dict(
        torch.load("/home/odeliam/PycharmProjects/Transferability/MNIST/log/fc1_new/mnist_clean/best-105.pth"))
    model2.load_state_dict(
        torch.load(
            "/home/odeliam/PycharmProjects/Transferability/MNIST/log/fc1_new/mnist_clean/best-105.pth"))

    # model2.load_state_dict(
    #     torch.load("/home/odeliam/PycharmProjects/Transferability/MNIST/log/fc1_new/projected/shafi_advtrain_testand1train_projected_pca32q_sameinit_100/best-1910.pth"))

    x = model2.fc1.weight.clone().detach()
    for i in range(256):
        x[i] = project_diff_on_global_basis(model2.fc1.weight[i])

    model2.fc1.weight = torch.nn.Parameter(x)
    # for name, parameter in model2.named_parameters():
    #     # print("bbbbbb", parameter)
    #     if "weight" in name:
    #         print("blablabla", name, parameter.shape)
    #         parameter[0] = project_diff_on_global_basis(parameter[0])

    model1, model2 = model1.eval(), model2.eval()
    model1, model2 = model1.cuda(), model2.cuda()

    encoder = MnistEncoder()

    on_global_manifold_angles = torch.tensor([]).cuda()
    off_global_manifold_angles = torch.tensor([]).cuda()

    on_global_onlocal_manifold_angles = torch.tensor([]).cuda()
    on_global_offlocal_manifold_angles = torch.tensor([]).cuda()

    # eps = 16/255
    # eps_iter = 2 * eps / 50
    eps = 1.0
    eps_iter = eps
    loss_fn = margin_loss

    test_set = dataset.MNIST.getTestSetIterator(batch_size=1)
    #

    attack1 = APGD(eps=eps, norm=2, targeted=False)

    for data, target in test_set:
        print("------------------------------")
        data, target = data.cuda(), target.cuda()
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

        adv_m1 = attack1.attack(model1, data, adv_target1)
        adv_m2 = attack1.attack(model2, data, adv_target2)

        print("angle between adversarial vectors : ", algebraTools.angle_between_tensor(adv_m1 - data, adv_m2 - data))

        adv1_diff = adv_m1 - data
        adv2_diff = adv_m2 - data

        if torch.norm(adv1_diff, p=2) == 0:
            continue



        def project_diff_on_local_on_global_basis(x):
            local_basis = torch.zeros((784, 784), dtype=torch.float).cuda()
            local_basis[:16, :] = projectionTools.find_local_manifold_around_image(encoder, data)

            global_basis = torch.zeros((784, 784), dtype=torch.float).cuda()
            global_basis[:model3.pca.k, :] = model3.pca.V.T[:model3.pca.k, :]

            first_proj = projectionTools.project_diff_on_basis(global_basis, x)
            return projectionTools.project_diff_on_basis(local_basis, first_proj)

        def project_diff_on_global_off_local_basis(x):
            global_basis = torch.zeros((784, 784), dtype=torch.float).cuda()
            global_basis[:model3.pca.k, :] = model3.pca.V.T[:model3.pca.k, :]
            # global_basis = create_orthonormal_basis(global_basis)

            local_basis = torch.eye(784).cuda()
            local_basis[:16, :] = projectionTools.find_local_manifold_around_image(encoder, data)
            full_local_basis = create_orthonormal_basis(local_basis)
            off_local_basis = full_local_basis
            off_local_basis[:16, :] = 0

            first_proj = projectionTools.project_diff_on_basis(global_basis, x)
            return projectionTools.project_diff_on_basis(off_local_basis, first_proj)

        adv_m1_on_global_diff = project_diff_on_global_basis(adv1_diff)
        adv_m2_on_global_diff = project_diff_on_global_basis(adv2_diff)
        adv_m1_proj_on_global = data + adv_m1_on_global_diff
        adv_m2_proj_on_global = data + adv_m2_on_global_diff

        print("on manifold distances:")
        print("on manifold adversarial distance 1:", torch.norm(adv_m1_proj_on_global - data, p=2))
        print("on manifold adversarial distance 2:", torch.norm(adv_m2_proj_on_global - data, p=2))
        on_manifold_angle = algebraTools.angle_between_tensor(adv_m1_proj_on_global - data, adv_m2_proj_on_global - data)
        print("on manifold angle between adversarial vectors:", on_manifold_angle)
        on_global_manifold_angles = torch.cat((on_global_manifold_angles, on_manifold_angle.unsqueeze(0)), dim=0)

        adv_m1_off_manifold = data + project_diff_off_global_basis(adv1_diff)
        adv_m2_off_manifold = data + project_diff_off_global_basis(adv2_diff)

        print("off manifold adversarial distance 1:", torch.norm(adv_m1_off_manifold - data, p=2))
        print("off manifold adversarial distance 2:", torch.norm(adv_m2_off_manifold - data, p=2))
        off_manifold_angle = algebraTools.angle_between_tensor(adv_m1_off_manifold - data, adv_m2_off_manifold - data)
        print("off manifold angle between adversarial vectors:", off_manifold_angle)
        off_global_manifold_angles = torch.cat((off_global_manifold_angles, off_manifold_angle.unsqueeze(0)), dim=0)

        adv_m1_proj_on_globalnlocal = data + project_diff_on_local_on_global_basis(adv1_diff)
        adv_m2_proj_on_globalnlocal = data + project_diff_on_local_on_global_basis(adv2_diff)

        print("on manifold distances:")
        print("on manifold adversarial distance 1:", torch.norm(adv_m1_proj_on_globalnlocal - data, p=2))
        print("on manifold adversarial distance 2:", torch.norm(adv_m2_proj_on_globalnlocal - data, p=2))
        on_manifold_angle = algebraTools.angle_between_tensor(adv_m1_proj_on_globalnlocal - data,
                                                              adv_m2_proj_on_globalnlocal - data)
        print("on manifold angle between adversarial vectors:", on_manifold_angle)
        on_global_onlocal_manifold_angles = torch.cat((on_global_onlocal_manifold_angles, on_manifold_angle.unsqueeze(0)), dim=0)

        # on_global_off_local_diff1 = adv_m1_on_global_diff - project_diff_on_local_basis(adv_m1_on_global_diff)
        # on_global_off_local_diff2 = adv_m2_on_global_diff - project_diff_on_local_basis(adv_m2_on_global_diff)

        adv_m1_on_global_offlocal = data + project_diff_on_global_off_local_basis(adv1_diff)
        adv_m2_on_global_offlocal = data + project_diff_on_global_off_local_basis(adv2_diff)

        print("off manifold adversarial distance 1:", torch.norm(adv_m1_on_global_offlocal - data, p=2))
        print("off manifold adversarial distance 2:", torch.norm(adv_m2_on_global_offlocal - data, p=2))
        off_manifold_angle = algebraTools.angle_between_tensor(adv_m1_on_global_offlocal - data, adv_m2_on_global_offlocal - data)
        print("off manifold angle between adversarial vectors:", off_manifold_angle)
        on_global_offlocal_manifold_angles = torch.cat((on_global_offlocal_manifold_angles, off_manifold_angle.unsqueeze(0)), dim=0)


        #TODO: see if the margin between classes is closing like linear function with cosine ratio!
        basicview.view_classification_changes_multirow([model1] * 4 + [model2]*4, [data] * 8,
                                                       [adv_m1, adv_m1_off_manifold, adv_m1_on_global_offlocal,
                                                        adv_m1_proj_on_globalnlocal, adv_m1, adv_m1_off_manifold,
                                                        adv_m1_on_global_offlocal, adv_m1_proj_on_globalnlocal],
                                                       txt=["model 1, adv 1", "model 1, adv 1 off global manifold",
                                                            "model 1 adv 1 on global, off local",
                                                            "model 1 adv 1 on global on local",
                                                            "model 2, adv 1",
                                                            "model 2, adv 1 off global manifold",
                                                            "model 2 adv 1 on global, off local",
                                                            "model 2 adv 1 on global on local"])
        # basicview.view_classification_changes_multirow([model2] * 5, [data] * 5,
        #                                                [adv_m1, adv_m1_proj_on, adv_m1_off_manifold, adv_m2_proj_on,
        #                                                 adv_m2_off_manifold],
        #                                                txt=["model 2, adv 1", "model 2, adv 1 on",
        #                                                     "model 2 adv 1 proj off", "model 2 adv 2 on ",
        #                                                     "model 2 adv2 proj off"])

        # basicview.view_classification_changes_multirow([model1]*5, [data]*5, [adv_m1, adv_m1_proj_on, adv_m1_off_manifold, adv_m2_proj_on, adv_m2_off_manifold],
        #                                                txt=["model 1, adv 1", "model 2, adv 1 on", "model 1 adv 1 proj off", "model 1 adv 2 on", "model 1 adv2 proj off"])
        # basicview.view_classification_changes_multirow([model2]*5, [data]*5, [adv_m1, adv_m1_proj_on, adv_m1_off_manifold, adv_m2_proj_on, adv_m2_off_manifold],
        #                                                txt=["model 2, adv 1", "model 2, adv 1 on", "model 2 adv 1 proj off", "model 2 adv 2 on ", "model 2 adv2 proj off"])

        # print("on manifold angle mean:", on_manifold_angles.mean())
        # print("off manifold angle mean: ", off_manifold_angles.mean())





def exp11():
    model1 = MLPClassifier()
    model2 = MLPClassifier()
    model3 = MLPnPCAClassifier(num_of_dims=16).cuda()

    model1.load_state_dict(
        torch.load("/home/odeliam/PycharmProjects/Transferability/MNIST/log/robust-fc100000/best-25.pth"))
    model2.load_state_dict(
        torch.load("/home/odeliam/PycharmProjects/Transferability/MNIST/log/non-robust-fc2/best-25.pth"))

    model1, model2 = model1.eval(), model2.eval()
    model1, model2 = model1.cuda(), model2.cuda()

    encoder = MnistEncoder()

    on_manifold_angles = torch.tensor([]).cuda()
    off_manifold_angles = torch.tensor([]).cuda()

    # eps = 16/255
    # eps_iter = 2 * eps / 50
    eps = 1.0
    eps_iter = eps
    loss_fn = margin_loss

    test_set = dataset.MNIST.getTestSetIterator(batch_size=1)
    #
    # attack1 = PGD.PGDAttack(predict=model1, eps=eps, nb_iter=50, eps_iter=eps_iter/50,
    #                        rand_init=False, ord=2, targeted=False, clip_min=0., clip_max=1.,
    #                        loss_fn=loss_fn)
    #
    # attack2 = PGD.PGDAttack(predict=model2, eps=eps, nb_iter=50, eps_iter=eps_iter/50,
    #                        rand_init=False, ord=2, targeted=False, clip_min=0., clip_max=1.,
    #                        loss_fn=loss_fn)

    attack1 = APGD(eps=eps, norm=2, targeted=False)
    attack2 = APGD(eps=eps, norm=2, targeted=False)

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

        # adv_m1 = attack1.perturb(data, adv_target1)
        # adv_m2 = attack2.perturb(data, adv_target2)
        adv_m1 = attack1.attack(model1, data, adv_target1)
        adv_m2 = attack1.attack(model2, data, adv_target2)

        print("angle between adversarial vectors : ", algebraTools.angle_between_tensor(adv_m1 - data, adv_m2 - data))

        adv1_diff = adv_m1 - data
        adv2_diff = adv_m2 - data

        basis = projectionTools.find_local_manifold_around_image(encoder, data)

        def project_diff_on_basis(x, basis):
            # return projectionTools.project_diff_on_basis(basis, x)

            # res = model1.pca.forward(x)[]
            basis1 = model3.pca.V[:, :model3.pca.k].T
            basis1 = create_orthonormal_basis(basis1)
            x_flat = torch.reshape(x, (x.shape[0], 28 * 28)).squeeze(0)
            return project_on_basis(basis1, x_flat).reshape_as(x)

        adv_m1_proj_on = data + project_diff_on_basis(adv1_diff, basis)
        adv_m2_proj_on = data + project_diff_on_basis(adv2_diff, basis)

        print("on manifold distances:")
        print("on manifold adversarial distance 1:", torch.norm(adv_m1_proj_on - data, p=2))
        print("on manifold adversarial distance 2:", torch.norm(adv_m2_proj_on - data, p=2))
        on_manifold_angle = algebraTools.angle_between_tensor(adv_m1_proj_on - data, adv_m2_proj_on - data)
        print("on manifold angle between adversarial vectors:", on_manifold_angle)
        on_manifold_angles = torch.cat((on_manifold_angles, on_manifold_angle.unsqueeze(0)), dim=0)

        # data_off_manifold = data - data_proj
        adv_m1_off_manifold = data + (adv1_diff - project_diff_on_basis(adv1_diff, basis))
        adv_m2_off_manifold = data + (adv2_diff - project_diff_on_basis(adv2_diff, basis))

        print("off manifold adversarial distance 1:", torch.norm(adv_m1_off_manifold - data, p=2))
        print("off manifold adversarial distance 2:", torch.norm(adv_m2_off_manifold - data, p=2))
        off_manifold_angle = algebraTools.angle_between_tensor(adv_m1_off_manifold - data, adv_m2_off_manifold - data)
        print("off manifold angle between adversarial vectors:", off_manifold_angle)
        off_manifold_angles = torch.cat((off_manifold_angles, off_manifold_angle.unsqueeze(0)), dim=0)

        #TODO: see if the margin between classes is closing like linear function with cosine ratio!

        basicview.view_classification_changes_multirow([model1]*5, [data]*5, [adv_m1, adv_m1_proj_on, adv_m1_off_manifold, adv_m2_proj_on, adv_m2_off_manifold],
                                                       txt=["model 1, adv 1", "model 2, adv 1 on", "model 1 adv 1 proj off", "model 1 adv 2 on", "model 1 adv2 proj off"])
        basicview.view_classification_changes_multirow([model2]*5, [data]*5, [adv_m1, adv_m1_proj_on, adv_m1_off_manifold, adv_m2_proj_on, adv_m2_off_manifold],
                                                       txt=["model 2, adv 1", "model 2, adv 1 on", "model 2 adv 1 proj off", "model 2 adv 2 on ", "model 2 adv2 proj off"])

        print("on manifold angle mean:", on_manifold_angles.mean())
        print("off manifold angle mean: ", off_manifold_angles.mean())


def exp1():
    model1 = MLPnPCAClassifier(num_of_dims=784)
    model2 = MLPnPCAClassifier(num_of_dims=784)

    model1.load_state_dict(torch.load("/home/odeliam/PycharmProjects/Transferability/MNIST/log/nonrobust-pcafull/best-25.pth"))
    model2.load_state_dict(torch.load("/home/odeliam/PycharmProjects/Transferability/MNIST/log/nonrobust-pcafull2/best-25.pth"))

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
            # res = model1.pca.forward(x)[]
            basis = model1.pca.V[:, :500].T
            x_flat = torch.reshape(x, (x.shape[0], 28 * 28)).squeeze(0)
            return project_on_basis(basis, x_flat).reshape_as(x)

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

        # #pca space angles
        # def project_diff_off_basis_sp(x):
        #     x_flat = torch.reshape(x, (x.shape[0], 28 * 28))
        #     zresult = torch.matmul(x_flat, model1.pca.V)
        #     res = torch.zeros_like(x)
        #     res.view(-1)[16:] = zresult.view(-1)[16:]
        #     return res
        #
        # adv1_diff_sp = model1.pre_process(adv1_diff)
        # adv2_diff_sp = model1.pre_process(adv2_diff)
        #
        # print("sp on manifold distances:")
        # print("sp on manifold adversarial distance 1:", torch.norm(adv1_diff_sp, p=2))
        # print("sp on manifold adversarial distance 2:", torch.norm(adv2_diff_sp, p=2))
        # on_manifold_angle = algebraTools.angle_between_tensor(adv1_diff_sp, adv2_diff_sp)
        # print("sp on manifold angle between adversarial vectors:", on_manifold_angle)
        # on_manifold_angles = torch.cat((on_manifold_angles, on_manifold_angle.unsqueeze(0)), dim=0)
        #
        # # data_off_manifold = data - data_proj
        # adv_m1_proj_off_manifold = project_diff_off_basis_sp(adv1_diff)
        # adv_m2_proj_off_manifold = project_diff_off_basis_sp(adv2_diff)
        #
        # print("sp off manifold adversarial distance 1:", torch.norm(adv_m1_proj_off_manifold, p=2))
        # print("sp off manifold adversarial distance 2:", torch.norm(adv_m2_proj_off_manifold, p=2))
        # off_manifold_angle = algebraTools.angle_between_tensor(adv_m1_proj_off_manifold, adv_m2_proj_off_manifold)
        # print("sp off manifold angle between adversarial vectors:", off_manifold_angle)
        # off_manifold_angles = torch.cat((off_manifold_angles, off_manifold_angle.unsqueeze(0)), dim=0)

        #TODO: see if the margin between classes is closing like linear function with cosine ratio!

        # basicview.view_classification_changes_multirow([model1] * 5, [data] * 5,
        #                                                [adv_m1, adv_m1_proj_on, adv_m1_off_manifold, adv_m2_proj_on,
        #                                                 adv_m2_off_manifold],
        #                                                txt=["model 1, adv 1", "model 1, adv 1 on",
        #                                                     "model 1 adv 1 proj off", "model 1 adv 2 on",
        #                                                     "model 1 adv2 proj off"])
        # basicview.view_classification_changes_multirow([model2] * 5, [data] * 5,
        #                                                [adv_m1, adv_m1_proj_on, adv_m1_off_manifold, adv_m2_proj_on,
        #                                                 adv_m2_off_manifold],
        #                                                txt=["model 2, adv 1", "model 2, adv 1 on",
        #                                                     "model 2 adv 1 proj off", "model 2 adv 2 on ",
        #                                                     "model 2 adv2 proj off"])

    print("on manifold angle mean:", on_manifold_angles.mean())
    print("off manifold angle mean: ", off_manifold_angles.mean())

def exp0():
    model1 = MLPnPCAClassifier(num_of_dims=16)
    model2 = MLPnPCAClassifier(num_of_dims=16)

    # model1.load_state_dict(torch.load("/home/odeliam/PycharmProjects/Transferability/MNIST/log/non-robust/init.pth"))
    # model2.load_state_dict(torch.load("/home/odeliam/PycharmProjects/Transferability/MNIST/log/non-robust/init.pth"))

    model1.load_state_dict(
        torch.load("/home/odeliam/PycharmProjects/Transferability/MNIST/log/non-robust-unormalized/best-35.pth"))
    model2.load_state_dict(
        torch.load("/home/odeliam/PycharmProjects/Transferability/MNIST/log/non-robust-unormalized2/best-35.pth"))

    model1, model2 = model1.eval(), model2.eval()
    model1, model2 = model1.cuda(), model2.cuda()

    on_manifold_angles = torch.tensor([]).cuda()
    off_manifold_angles = torch.tensor([]).cuda()

    # eps = 16/255
    # eps_iter = 2 * eps / 50
    eps = 1.5
    eps_iter = eps
    loss_fn = margin_loss

    test_set = dataset.MNIST.getTestSetIterator(batch_size=1)

    attack1 = PGD.PGDAttack(predict=model1.mlp, eps=eps, nb_iter=1, eps_iter=eps_iter,
                           rand_init=False, ord=2, targeted=False, clip_min=-700., clip_max=700.,
                           loss_fn=loss_fn)

    attack2 = PGD.PGDAttack(predict=model2.mlp, eps=eps, nb_iter=1, eps_iter=eps_iter,
                           rand_init=False, ord=2, targeted=False, clip_min=-700., clip_max=700.,
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

        projected_data = model1.pre_process(data)

        model1_classification_mlp = model1.mlp(projected_data).data.max(1)[1]
        model2_classification_mlp = model2.mlp(projected_data).data.max(1)[1]
        print("mlp current classification by 1: {}, by 2: {}".format(model1_classification_mlp, model2_classification_mlp))

        adv_m1 = attack1.perturb(projected_data)
        adv_m2 = attack2.perturb(projected_data)

        print("adversarial distances: ", torch.norm(adv_m1-projected_data, p=2), torch.norm(adv_m2-projected_data, p=2))
        print("angle between adversarial vectors : ", algebraTools.angle_between_tensor(adv_m1 - projected_data, adv_m2 - projected_data))
        # print("angle between adversarial vectors - the diff norm : ", torch.norm((adv_m1 - adv_m2), p=2))

        # encoded_image = encoder.encode_and_decode(data)

        adv1_diff = adv_m1 - projected_data
        adv2_diff = adv_m2 - projected_data

        def project_diff_on_basis(x):
            res = torch.zeros_like(x)
            res.view(-1)[:16] = x.view(-1)[:16]
            return res

        adv_m1_proj_on = projected_data + project_diff_on_basis(adv1_diff)
        adv_m2_proj_on = projected_data + project_diff_on_basis(adv2_diff)

        print("on manifold distances:")
        print("on manifold adversarial distance 1:", torch.norm(adv_m1_proj_on - projected_data, p=2))
        print("on manifold adversarial distance 2:", torch.norm(adv_m2_proj_on - projected_data, p=2))
        on_manifold_angle = algebraTools.angle_between_tensor(adv_m1_proj_on - projected_data, adv_m2_proj_on - projected_data)
        print("on manifold angle between adversarial vectors:", on_manifold_angle)
        on_manifold_angles = torch.cat((on_manifold_angles, on_manifold_angle.unsqueeze(0)), dim=0)

        # projected_data_off_manifold = projected_data - projected_data_proj
        adv_m1_off_manifold = projected_data + (adv1_diff - project_diff_on_basis(adv1_diff))
        adv_m2_off_manifold = projected_data + (adv2_diff - project_diff_on_basis(adv2_diff))

        print("off manifold adversarial distance 1:", torch.norm(adv_m1_off_manifold - projected_data, p=2))
        print("off manifold adversarial distance 2:", torch.norm(adv_m2_off_manifold - projected_data, p=2))
        off_manifold_angle = algebraTools.angle_between_tensor(adv_m1_off_manifold - projected_data, adv_m2_off_manifold - projected_data)
        print("off manifold angle between adversarial vectors:", off_manifold_angle)
        off_manifold_angles = torch.cat((off_manifold_angles, off_manifold_angle.unsqueeze(0)), dim=0)

        #TODO: see if the margin between classes is closing like linear function with cosine ratio!

        # basicview.view_classification_changes_multirow([model1.mlp]*5, [projected_data]*5, [adv_m1, adv_m1_proj_on, adv_m1_off_manifold, adv_m2_proj_on, adv_m2_off_manifold],
        #                                                txt=["model 1, adv 1", "model 1, adv 1 on", "model 1 adv 1 proj off", "model 1 adv 2 on", "model 1 adv2 proj off"])
        # basicview.view_classification_changes_multirow([model2.mlp]*5, [projected_data]*5, [adv_m2, adv_m1_proj_on, adv_m1_off_manifold, adv_m2_proj_on, adv_m2_off_manifold],
        #                                                txt=["model 2, adv 1", "model 2, adv 1 on", "model 2 adv 1 proj off", "model 2 adv 2 on", "model 2 adv2 proj off"])

    print("on manifold angle mean:", on_manifold_angles.mean())
    print("off manifold angle mean: ", off_manifold_angles.mean())


def distances():
    totals = torch.zeros(10, 10).cuda()
    totals_count = torch.zeros(10, 10).cuda()
    train_set = dataset.MNIST.getTestSetIterator(batch_size=1)
    train_set2 = dataset.MNIST.getTestSetIterator(batch_size=1)
    for data, target in train_set:
        data, target = data.cuda(), target.cuda()
        for data2, target2 in train_set2:
            data2, target2 = data2.cuda(), target2.cuda()
            if target != target2:
                totals[target, target2] += torch.norm(data-data2, p=2)
                totals_count[target, target2] += 1

    print("distances:", totals / totals_count)




if __name__ == '__main__':
    # distances()
    # exp0()
    # exp1()
    # exp11()
    exp12()
    # exp2()
    # exp3()