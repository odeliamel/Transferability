import sys
from trials.utils import *
sys.path.insert(1, '/home/odeliam/PycharmProjects/EBT')
from models.classifiers.cifar import cifar_manager
from attacks.losses import margin_loss
import attacks.EBT.getTarget as getTarget
# import advertorch.attacks.iterative_projected_gradient as PGD
import trials.local_pgd as PGD
from datasets import dataset
from algebra import algebraTools
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from views import basicview
from algebra.linearApproximation import find_matrix_around_image_batch
from models.autoencoders.cifar import cifar_encoder_manager
from algebra.projectionTools import find_local_manifold_around_image, project_diff_on_basis, project_diff_off_basis
from algebra.projectionTools import get_random_manifold_basis
from trials.transferability_tst import check_transferability


from threadpoolctl import threadpool_limits
_thread_limit = threadpool_limits(limits=8)
print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def general_transferability():
    mang1 = cifar_manager.load_model(model_name="vgg16", exp_name="cifar-sgd-new-nopretrain260-3")
    model1 = mang1.model
    # mang1.test()

    # mang2 = cifar_manager.load_model(model_name="vgg16", exp_name="cifar-sgd-new-nopretrain260-3")
    mang2 = cifar_manager.load_model(model_name="vgg16", exp_name="cifar-sgd-new-nopretrain260-2")
    # mang2 = cifar_manager.load_model(model_name="resnext", exp_name="cifar-resnext")
    model2 = mang2.model
    # mang2.test()

    model1, model2 = model1.eval(), model2.eval()
    model1, model2 = model1.cuda(), model2.cuda()

    iters = 50
    # eps = 8/255
    # eps = 0.5
    eps = 1.0
    eps_iter = eps / iters
    loss_fn = margin_loss

    test_set = dataset.CIFARData.getTestSetIterator(batch_size=40)

    attack1 = PGD.PGDAttack(predict=model1, eps=eps, nb_iter=iters, eps_iter=eps_iter,
                            rand_init=False, ord=2, targeted=False, clip_min=0., clip_max=1.,
                            loss_fn=loss_fn)

    find_target = lambda data, target: target
    # find_target = lambda data, target:getTarget.get_best_target(model1, data, target, epsilon=eps, ord=float("inf"),
    #                                            loss_fn=loss_fn)

    check_transferability(model1=mang1, model2=mang2, attack=attack1, test_loader=test_set, target_fn=find_target)

def exp1():
    # mang1 = cifar_manager.load_model(model_name="vgg16", exp_name="cifar-sgd-new-nopretrain260-3")
    mang1 = cifar_manager.load_model(model_name="simple", exp_name="cifar-simple")
    model1 = mang1.model
    # mang1.test()

    # mang2 = cifar_manager.load_model(model_name="vgg16", exp_name="cifar-sgd-new-nopretrain260-2")
    # mang2 = cifar_manager.load_model(model_name="resnext", exp_name="cifar-resnext")
    mang2 = cifar_manager.load_model(model_name="simple", exp_name="cifar-simple")
    model2 = mang2.model
    # mang2.test()

    model1, model2 = model1.eval(), model2.eval()
    model1, model2 = model1.cuda(), model2.cuda()

    iters = 1
    # eps = 8/255
    eps = 0.5
    # eps = 1.0
    eps_iter = eps/iters
    loss_fn = margin_loss

    test_set = dataset.CIFARData.getTestSetIterator(batch_size=1)

    attack1 = PGD.PGDAttack(predict=model1, eps=eps, nb_iter=iters, eps_iter=eps_iter,
                           rand_init=False, ord=2, targeted=True, clip_min=0., clip_max=1.,
                           loss_fn=loss_fn)

    attack2 = PGD.PGDAttack(predict=model2, eps=eps, nb_iter=iters, eps_iter=eps_iter,
                           rand_init=True, ord=2, targeted=True, clip_min=0., clip_max=1.,
                           loss_fn=loss_fn)

    for data, target in test_set:
        print("----------------------------------------------------------------------------------------")
        data, target = data.cuda(), target.cuda()

        adv_target1 = getTarget.get_best_target(model1, data, target, epsilon=eps, ord=2,
                                               loss_fn=loss_fn)
        adv_target2 = getTarget.get_best_target(model2, data, target, epsilon=eps, ord=2,
                                                loss_fn=loss_fn)

        # adv_target1 = target
        # adv_target2 = target
        print("original target: ", target.item())
        model1_classification = model1(data).data.max(1)[1]
        model2_classification = model2(data).data.max(1)[1]
        print("current classification by 1: {}, by 2: {}".format(model1_classification, model2_classification))
        print("suggested best targets: 1 - {}, 2 - {}".format(adv_target1.item(), adv_target2.item()))

        if model1_classification.item() != model2_classification.item()\
                or model1_classification.item() != target.item() or model2_classification.item() != target.item():
            continue

        if adv_target1.item() != adv_target2.item():
            print("****different best targets, choosing the first - {} ****".format(adv_target1))
            adv_target2 = adv_target1.clone()

        adv_img1 = attack1.perturb(data, adv_target1)
        adv_m1 = adv_img1 - data

        adv_img2 = attack2.perturb(data, adv_target2)
        adv_m2 = adv_img2 - data

        print("angle between adversarial vectors: ", algebraTools.angle_between_tensor(adv_m1, adv_m2))
        print("the diff norm between adversarial images: ", torch.norm((adv_img1 - adv_img2), p=2))
        print("the Linf distance to the original image: 1: {}, 2: {}".format(torch.norm(adv_m1, p=float("inf")), torch.norm(adv_m2, p=float("inf"))))
        print("the L2 distance to the original image: 1: {}, 2: {}".format(torch.norm(adv_m1), torch.norm(adv_m2)))

        # ----------- projection on the local adversarial vectors -------------
        print("\n++++++++ projection on the local adversarial vectors ++++++++++++")
        linmatrix1 = find_matrix_around_image_batch(data, 0.0, model=model1)[0]
        linmatrix2 = find_matrix_around_image_batch(data, 0.0, model=model2)[0]

        comb_linmatrix = torch.cat([linmatrix1[target], linmatrix1[adv_target1]])

        print("linear matrices relation between the different lines")
        for i in range(10):
            print("lines {}, angle between {}".format(i, algebraTools.angle_between_tensor(linmatrix1[i], linmatrix2[i])))

        print("adv norm: 1: {}, 2: {}".format(torch.norm(adv_m1, p=2), torch.norm(adv_m2, p=2)))
        basis1 = algebraTools.create_orthonormal_basis(linmatrix1)
        basis2 = algebraTools.create_orthonormal_basis(comb_linmatrix)


        # first lin space
        print("\n$$ first linear space $$")
        adv1_proj1 = algebraTools.project_on_basis(basis1, adv_m1.reshape(3*32*32))
        adv2_proj1 = algebraTools.project_on_basis(basis1, adv_m2.reshape(3*32*32))
        print("angle between the 2 projections of adv on lin1:{}".format(algebraTools.angle_between_tensor(adv1_proj1, adv2_proj1)))
        print("projections norms: adv1: {} adv2: {}".format(torch.norm(adv1_proj1, p=2), torch.norm(adv2_proj1, p=2)))
        print("the diff norm between projected adversarial images: ", torch.norm((adv1_proj1 - adv2_proj1), p=2))
        print("by model 1 - adv1 effect: {} ({}/{}), adv2 effect: {} ({}/{})".format(
            *get_relative_effect(data, model1, adv_m1, adv1_proj1.reshape_as(adv_m1), target, adv_target1),
            *get_relative_effect(data, model1, adv_m1, adv2_proj1.reshape_as(adv_m2), target, adv_target1)
        ))

        # second linear space
        print("\n$$ second linear space $$")
        adv1_proj2 = algebraTools.project_on_basis(basis2, adv_m1.reshape(3*32*32))
        adv2_proj2 = algebraTools.project_on_basis(basis2, adv_m2.reshape(3*32*32))
        print("angle between the 2 projection of adv on lin2:{}".format(
            algebraTools.angle_between_tensor(adv1_proj2, adv2_proj2)))
        print("projections norms: adv1: {} adv2: {}".format(torch.norm(adv1_proj2, p=2), torch.norm(adv2_proj2, p=2)))
        print("the diff norm between projected adversarial images: ", torch.norm((adv2_proj2 - adv1_proj2), p=2))
        print("by model 2 - adv1 effect: {}, adv2 effect: {}".format(
            get_relative_effect(data, model2, adv_m1, adv1_proj2.reshape_as(adv_m1), target, adv_target1),
            get_relative_effect(data, model2, adv_m2, adv2_proj2.reshape_as(adv_m2), target, adv_target1)))

        # random linear space
        print("\n$$ random linear space of dimension {} $$".format(linmatrix1.shape[0]))
        rand_basis = get_random_manifold_basis(image_height=data.shape[-1], manifold_dim=linmatrix1.shape[0])
        adv1_projrand = algebraTools.project_on_basis(rand_basis, adv_m1.reshape(3 * 32 * 32))
        adv2_projrand = algebraTools.project_on_basis(rand_basis, adv_m2.reshape(3 * 32 * 32))
        print("angle between the 2 projection of adv on lin2:{}".format(
            algebraTools.angle_between_tensor(adv1_projrand, adv2_projrand)))
        print("projections norms: adv1: {} adv2: {}".format(torch.norm(adv1_projrand, p=2), torch.norm(adv2_projrand, p=2)))
        print("the diff norm between projected adversarial images: ", torch.norm((adv1_projrand - adv2_projrand), p=2))
        print("by model 1 - adv1 effect: {} ({}/{}), adv2 effect: {} ({}/{})".format(
            *get_relative_effect(data, model1, adv_m1, adv1_projrand.reshape_as(adv_m1), target, adv_target1),
            *get_relative_effect(data, model1, adv_m1, adv2_projrand.reshape_as(adv_m2), target, adv_target1)
        ))


        # ----------- projection on a low dimentional manifold -------------
        print("\n++++++++ projection on the low dimensional image manifold ++++++++++++")
        encoder = cifar_encoder_manager.load_model(model_name="vgg16", exp_name="cifar-all-nopretrained")
        enc_basis = find_local_manifold_around_image(encoder=encoder, image=data)

        on_proj1 = project_diff_on_basis(enc_basis, adv_m1)
        on_proj2 = project_diff_on_basis(enc_basis, adv_m2)

        off_proj1 = project_diff_off_basis(enc_basis, adv_m1)
        off_proj2 = project_diff_off_basis(enc_basis, adv_m2)

        print("on manifold projected vector's norm: adv1: {}, adv2: {}".format(torch.norm(on_proj1, p=2), torch.norm(on_proj2, p=2)))
        print("angle between the adv. vectors projected on-manifold: {}". format(
            algebraTools.angle_between_tensor(on_proj1, on_proj2)))
        print("off manifold projected vector's norm: adv1: {}, adv2: {}".format(torch.norm(off_proj1, p=2), torch.norm(off_proj2, p=2)))
        print("angle between the adv. vectors projected off-manifold: {}".format(
            algebraTools.angle_between_tensor(off_proj1, off_proj2)))

        # projection on a random low dimensional manifold
        print("\n++ projection on a random low dimensional manifold of dimension {}++".format(enc_basis.shape[0]))
        rand_basis = get_random_manifold_basis(image_height=data.shape[-1], manifold_dim=enc_basis.shape[0])
        on_rand_proj1 = project_diff_on_basis(rand_basis, adv_m1)
        on_rand_proj2 = project_diff_on_basis(rand_basis, adv_m2)

        off_rand_proj1 = project_diff_off_basis(rand_basis, adv_m1)
        off_rand_proj2 = project_diff_off_basis(rand_basis, adv_m2)

        print("on manifold projected vector's norm: adv1: {}, adv2: {}".format(torch.norm(on_rand_proj1, p=2), torch.norm(on_rand_proj2, p=2)))
        print("angle between the adv. vectors projected on-manifold: {}".format(
            algebraTools.angle_between_tensor(on_rand_proj1, on_rand_proj2)))
        print("off manifold projected vector's norm: adv1: {}, adv2: {}".format(torch.norm(off_rand_proj1, p=2), torch.norm(off_rand_proj2, p=2)))
        print("angle between the adv. vectors projected off-manifold: {}".format(
            algebraTools.angle_between_tensor(off_rand_proj1, off_rand_proj2)))

        print("target line from the 2 linear matrices")
        project_on_off_manifold(enc_basis, rand_basis, linmatrix1[target.item()], linmatrix2[target.item()])

        print("adv vectors projected on 1 10-dim linear space")
        project_on_off_manifold(enc_basis, rand_basis, adv1_proj1, adv2_proj1)
        print("adv vectors projected on 2 10-dim linear space")
        project_on_off_manifold(enc_basis, rand_basis, adv1_proj2, adv2_proj2)

        basicview.view_classification_changes_multirow([mang1, mang2], [data, data], [adv_img1, adv_img1])
        basicview.view_classification_changes_multirow([mang1, mang2], [data, data], [adv_img2, adv_img2])
        # basicview.view_classification_changes_multirow(mang1, [data], [adv_m2])
        # basicview.view_classification_changes_multirow(mang2, [data], [adv_m2])



def exp2():
    mang1 = cifar_manager.load_model(model_name="vgg16", exp_name="cifar-sgd-new-nopretrain260-3")
    model1 = mang1.model
    # mang1.test()

    # mang2 = cifar_manager.load_model(model_name="vgg16", exp_name="cifar-sgd-new-nopretrain260-2")
    # mang2 = cifar_manager.load_model(model_name="resnext", exp_name="cifar-resnext")
    mang2 = cifar_manager.load_model(model_name="simple", exp_name="cifar-simple")
    model2 = mang2.model
    # mang2.test()

    model1, model2 = model1.eval(), model2.eval()
    model1, model2 = model1.cuda(), model2.cuda()

    iters = 20
    # eps = 8/255
    eps = 0.5
    # eps = 1.0
    eps_iter = eps/iters
    loss_fn = margin_loss
    targeted = True

    test_set = dataset.CIFARData.getTestSetIterator(batch_size=1)

    attack1 = PGD.PGDAttack(predict=model1, eps=eps, nb_iter=iters, eps_iter=eps_iter,
                           rand_init=False, ord=2, targeted=targeted, clip_min=0., clip_max=1.,
                           loss_fn=loss_fn)

    attack2 = PGD.PGDAttack(predict=model2, eps=eps, nb_iter=iters, eps_iter=eps_iter,
                           rand_init=False, ord=2, targeted=targeted, clip_min=0., clip_max=1.,
                           loss_fn=loss_fn)
    for data, target in test_set:
        print("----------------------------------------------------------------------------------------")
        data, target = data.cuda(), target.cuda()

        adv_target1 = getTarget.get_best_target(model1, data, target, epsilon=eps, ord=2,
                                               loss_fn=loss_fn)
        adv_target2 = getTarget.get_best_target(model2, data, target, epsilon=eps, ord=2,
                                                loss_fn=loss_fn)

        # adv_target1 = target
        # adv_target2 = target

        print("original target: ", target.item())
        model1_classification = model1(data).data.max(1)[1]
        model2_classification = model2(data).data.max(1)[1]
        print("current classification by 1: {}, by 2: {}".format(model1_classification, model2_classification))
        print("suggested best targets: 1 - {}, 2 - {}".format(adv_target1.item(), adv_target2.item()))

        if model1_classification.item() != model2_classification.item()\
                or model1_classification.item() != target.item() or model2_classification.item() != target.item():
            continue

        if adv_target1.item() != adv_target2.item():
            print("****different best targets, choosing the first - {} ****".format(adv_target1))
            adv_target2 = adv_target1.clone()

        adv_img1 =  attack1.perturb(data, adv_target1)
        adv_m1 = adv_img1 - data

        adv_img2 = attack2.perturb(data, adv_target2)
        adv_m2 = adv_img2 - data

        print("angle between adversarial vectors: ", algebraTools.angle_between_tensor(adv_m1, adv_m2))
        print("the diff norm between adversarial images: ", torch.norm((adv_img1 - adv_img2), p=2))
        print("the Linf distance to the original image: 1: {}, 2: {}".format(torch.norm(adv_m1, p=float("inf")), torch.norm(adv_m2, p=float("inf"))))
        print("the L2 distance to the original image: 1: {}, 2: {}".format(torch.norm(adv_m1), torch.norm(adv_m2)))

        # ----------- projection on the local adversarial vectors -------------
        print("\n++++++++ projection on the local adversarial vectors ++++++++++++")
        linmatrix1 = find_matrix_around_image_batch(data, 0.01, model=model1)[0]
        linmatrix2 = find_matrix_around_image_batch(data, 0.01, model=model2)[0]

        print("linear matrices relation between the different lines")
        for i in range(10):
            print("lines {}, angle between {}".format(i, algebraTools.angle_between_tensor(linmatrix1[i], linmatrix2[i])))


        print("adv norm: 1: {}, 2: {}".format(torch.norm(adv_m1, p=2), torch.norm(adv_m2, p=2)))
        basis1 = algebraTools.create_orthonormal_basis(linmatrix1)
        basis2 = algebraTools.create_orthonormal_basis(linmatrix2)

        # first lin space
        print("\n$$ first linear space $$")
        adv1_proj1 = algebraTools.project_on_basis(basis1, adv_m1.reshape(3*32*32))
        adv2_proj1 = algebraTools.project_on_basis(basis1, adv_m2.reshape(3*32*32))

        print("angle between the 2 projections of adv on lin1:{}".format(
            algebraTools.angle_between_tensor(adv1_proj1, adv2_proj1)))
        print("projections norms: adv1: {} adv2: {}".format(torch.norm(adv1_proj1, p=2), torch.norm(adv2_proj1, p=2)))
        print("by model 1 - adv1 effect: {}, adv2 effect: {}".format(
            get_relative_effect(data, model1, adv_m1, adv1_proj1.reshape_as(adv_m1), target, adv_target1),
            get_relative_effect(data, model1, adv_m2, adv2_proj1.reshape_as(adv_m2), target, adv_target1)
        ))

        # second linear space
        print("\n$$ second linear space $$")
        adv1_proj2 = algebraTools.project_on_basis(basis2, adv_m1.reshape(3*32*32))
        adv2_proj2 = algebraTools.project_on_basis(basis2, adv_m2.reshape(3*32*32))

        print("angle between the 2 projection of adv on lin2:{}".format(
            algebraTools.angle_between_tensor(adv1_proj2, adv2_proj2)))
        print("projections norms: adv1: {} adv2: {}".format(torch.norm(adv1_proj2, p=2), torch.norm(adv2_proj2, p=2)))
        print("by model 2 - adv1 effect: {}, adv2 effect: {}".format(
            get_relative_effect(data, model2, adv_m1, adv1_proj2.reshape_as(adv_m1), target, adv_target1),
            get_relative_effect(data, model2, adv_m2, adv2_proj2.reshape_as(adv_m2), target, adv_target1)))

        # random linear space
        print("\n$$ random linear space of dimension {} $$".format(linmatrix1.shape[0]))
        rand_basis = get_random_manifold_basis(image_height=data.shape[-1], manifold_dim=linmatrix1.shape[0])
        adv1_projrand = algebraTools.project_on_basis(rand_basis, adv_m1.reshape(3 * 32 * 32))
        adv2_projrand = algebraTools.project_on_basis(rand_basis, adv_m2.reshape(3 * 32 * 32))
        print("angle between the 2 projection of adv on lin2:{}".format(
            algebraTools.angle_between_tensor(adv1_projrand, adv2_projrand)))
        print("projections norms: adv1: {} adv2: {}".format(torch.norm(adv1_projrand, p=2), torch.norm(adv2_projrand, p=2)))
        print("by model 1 - adv1 effect: {}, adv2 effect: {}".format(
            get_relative_effect(data, model1, adv_m1, adv1_projrand.reshape_as(adv_m1), target, adv_target1),
            get_relative_effect(data, model1, adv_m2, adv2_projrand.reshape_as(adv_m2), target, adv_target1)))
        print("by model 2 - adv1 effect: {}, adv2 effect: {}".format(
            get_relative_effect(data, model2, adv_m1, adv1_projrand.reshape_as(adv_m1), target, adv_target1),
            get_relative_effect(data, model2, adv_m2, adv2_projrand.reshape_as(adv_m2), target, adv_target1)))




        # ----------- projection on a low dimentional manifold -------------
        print("\n++++++++ projection on the low dimensional image manifold ++++++++++++")
        encoder = cifar_encoder_manager.load_model(model_name="vgg16", exp_name="cifar-all-nopretrained")
        enc_basis = find_local_manifold_around_image(encoder=encoder, image=data)

        print("projecting 10 classes vectors (model1 matrix):")
        for i in range(10):
            i_proj = project_diff_on_basis(enc_basis, linmatrix1[i])
            print("On manifold norm of line {} : {}".format(i, get_relative_norm(linmatrix1[i], i_proj)))
            if i != target.item():
                i_diff_proj = project_diff_on_basis(enc_basis, linmatrix1[i]- linmatrix1[target])
                print("On manifold norm of diff line {} and target: {}".format(i, get_relative_norm(linmatrix1[i]-linmatrix1[target], i_diff_proj)))

        print("projecting 10 classes vectors (model2 matrix):")
        for i in range(10):
            i_proj = project_diff_on_basis(enc_basis, linmatrix2[i])
            print("On manifold norm of line {} : {}".format(i, get_relative_norm(linmatrix2[i], i_proj)))
            if i != target.item():
                i_diff_proj = project_diff_on_basis(enc_basis, linmatrix2[i]- linmatrix2[target])
                print("On manifold norm of diff line {} and target: {}".format(i, get_relative_norm(linmatrix2[i]-linmatrix2[target], i_diff_proj)))

        basicview.view_classification_changes_multirow([mang1, mang2], [data, data], [adv_img1, adv_img1])
        basicview.view_classification_changes_multirow([mang1, mang2], [data, data], [adv_img2, adv_img2])

def exp3():
    mang1 = cifar_manager.load_model(model_name="vgg16", exp_name="cifar-sgd-new-nopretrain260-3")
    model1 = mang1.model
    # mang1.test()

    # mang2 = cifar_manager.load_model(model_name="vgg16", exp_name="cifar-sgd-new-nopretrain260-2")
    # mang2 = cifar_manager.load_model(model_name="resnext", exp_name="cifar-resnext")
    mang2 = cifar_manager.load_model(model_name="simple", exp_name="cifar-simple")
    model2 = mang2.model
    # mang2.test()

    model1, model2 = model1.eval(), model2.eval()
    model1, model2 = model1.cuda(), model2.cuda()

    iters = 20
    # eps = 8/255
    eps = 0.5
    # eps = 1.0
    eps_iter = eps / iters
    loss_fn = margin_loss
    targeted = True

    test_set = dataset.CIFARData.getTestSetIterator(batch_size=1)

    attack1 = PGD.PGDAttack(predict=model1, eps=eps, nb_iter=iters, eps_iter=eps_iter,
                            rand_init=False, ord=2, targeted=targeted, clip_min=0., clip_max=1.,
                            loss_fn=loss_fn)

    attack2 = PGD.PGDAttack(predict=model2, eps=eps, nb_iter=iters, eps_iter=eps_iter,
                            rand_init=False, ord=2, targeted=targeted, clip_min=0., clip_max=1.,
                            loss_fn=loss_fn)
    for data, target in test_set:
        print("----------------------------------------------------------------------------------------")
        data, target = data.cuda(), target.cuda()

        adv_target1 = getTarget.get_best_target(model1, data, target, epsilon=eps, ord=2,
                                                loss_fn=loss_fn)
        adv_target2 = getTarget.get_best_target(model2, data, target, epsilon=eps, ord=2,
                                                loss_fn=loss_fn)

        # adv_target1 = target
        # adv_target2 = target

        print("original target: ", target.item())
        model1_classification = model1(data).data.max(1)[1]
        model2_classification = model2(data).data.max(1)[1]
        print("current classification by 1: {}, by 2: {}".format(model1_classification, model2_classification))
        print("suggested best targets: 1 - {}, 2 - {}".format(adv_target1.item(), adv_target2.item()))

        if model1_classification.item() != model2_classification.item() \
                or model1_classification.item() != target.item() or model2_classification.item() != target.item():
            continue

        if adv_target1.item() != adv_target2.item():
            print("****different best targets, choosing the first - {} ****".format(adv_target1))
            adv_target2 = adv_target1.clone()

        adv_img1 = attack1.perturb(data, adv_target1)
        adv_m1 = adv_img1 - data

        adv_img2 = attack2.perturb(data, adv_target2)
        adv_m2 = adv_img2 - data

        print("angle between adversarial vectors: ", algebraTools.angle_between_tensor(adv_m1, adv_m2))
        print("the L2 distance to the original image: 1: {}, 2: {}".format(torch.norm(adv_m1), torch.norm(adv_m2)))

        # ----------- projection on the local adversarial vectors -------------
        print("\n++++++++ projection on the local adversarial vectors ++++++++++++")
        linmatrix1 = find_matrix_around_image_batch(data, 0.01, model=model1)[0]
        linmatrix2 = find_matrix_around_image_batch(data, 0.01, model=model2)[0]

        basis1 = algebraTools.create_orthonormal_basis(linmatrix1)
        basis2 = algebraTools.create_orthonormal_basis(linmatrix2)

        # first lin space
        print("\n$$ first linear space $$")
        adv1_proj1 = algebraTools.project_on_basis(basis1, adv_m1.reshape(3 * 32 * 32))
        adv2_proj1 = algebraTools.project_on_basis(basis1, adv_m2.reshape(3 * 32 * 32))

        print("angle between the 2 projections of adv on lin1:{}".format(
            algebraTools.angle_between_tensor(adv1_proj1, adv2_proj1)))

        # second linear space
        print("\n$$ second linear space $$")
        adv1_proj2 = algebraTools.project_on_basis(basis2, adv_m1.reshape(3 * 32 * 32))
        adv2_proj2 = algebraTools.project_on_basis(basis2, adv_m2.reshape(3 * 32 * 32))

        print("angle between the 2 projection of adv on lin2:{}".format(
            algebraTools.angle_between_tensor(adv1_proj2, adv2_proj2)))

        # random linear space
        print("\n$$ random linear space of dimension {} $$".format(linmatrix1.shape[0]))
        rand_basis = get_random_manifold_basis(image_height=data.shape[-1], manifold_dim=linmatrix1.shape[0])
        adv1_projrand = algebraTools.project_on_basis(rand_basis, adv_m1.reshape(3 * 32 * 32))
        adv2_projrand = algebraTools.project_on_basis(rand_basis, adv_m2.reshape(3 * 32 * 32))
        print("angle between the 2 projection of adv on lin2:{}".format(
            algebraTools.angle_between_tensor(adv1_projrand, adv2_projrand)))

        # ----------- projection on a low dimentional manifold -------------
        print("\n++++++++ projection on the low dimensional image manifold ++++++++++++")
        encoder = cifar_encoder_manager.load_model(model_name="vgg16", exp_name="cifar-all-nopretrained")
        enc_basis = find_local_manifold_around_image(encoder=encoder, image=data)

        adv1_on = project_diff_on_basis(enc_basis, adv_m1)
        adv2_on = project_diff_on_basis(enc_basis, adv_m2)

        adv1_off = adv_m1 - adv1_on
        adv2_off = adv_m2 - adv2_on

        print("angle between the 2 projection of adv on manifold:{}".format(
            algebraTools.angle_between_tensor(adv1_on, adv2_on)))

        print("angle between the 2 projection of adv off manifold:{}".format(
            algebraTools.angle_between_tensor(adv1_off, adv2_off)))

        print("\n++ projection on a random low dimensional manifold of dimension {}++".format(enc_basis.shape[0]))
        rand_basis = get_random_manifold_basis(image_height=data.shape[-1], manifold_dim=enc_basis.shape[0])
        adv1_on_rand = project_diff_on_basis(rand_basis, adv_m1)
        adv2_on_rand = project_diff_on_basis(rand_basis, adv_m2)

        adv1_off_rand = adv_m1 - adv1_on_rand
        adv2_off_rand = adv_m2 - adv2_on_rand

        print("angle between the 2 projection of adv on random manifold:{}".format(
            algebraTools.angle_between_tensor(adv1_on_rand, adv2_on_rand)))

        print("angle between the 2 projection of adv off random manifold:{}".format(
            algebraTools.angle_between_tensor(adv1_off_rand, adv2_off_rand)))

        basicview.view_classification_changes_multirow([mang1, mang2], [data, data], [adv_img1, adv_img1])
        basicview.view_classification_changes_multirow([mang1, mang2], [data, data], [adv_img2, adv_img2])


def exp4():
    mang1 = cifar_manager.load_model(model_name="vgg16", exp_name="cifar-sgd-new-nopretrain260-3")
    model1 = mang1.model
    # mang1.test()

    mang2 = cifar_manager.load_model(model_name="vgg16", exp_name="cifar-sgd-new-nopretrain260-2")
    # mang2 = cifar_manager.load_model(model_name="resnext", exp_name="cifar-resnext")
    # mang2 = cifar_manager.load_model(model_name="simple", exp_name="cifar-simple")
    model2 = mang2.model
    # mang2.test()

    model1, model2 = model1.eval(), model2.eval()
    model1, model2 = model1.cuda(), model2.cuda()

    iters = 20
    # eps = 8/255
    eps = 0.5
    # eps = 1.0
    eps_iter = eps / iters
    loss_fn = margin_loss
    targeted = True

    test_set = dataset.CIFARData.getTestSetIterator(batch_size=1)

    attack1 = PGD.PGDAttack(predict=model1, eps=eps, nb_iter=iters, eps_iter=eps_iter,
                            rand_init=False, ord=2, targeted=targeted, clip_min=0., clip_max=1.,
                            loss_fn=loss_fn)

    attack2 = PGD.PGDAttack(predict=model2, eps=eps, nb_iter=iters, eps_iter=eps_iter,
                            rand_init=False, ord=2, targeted=targeted, clip_min=0., clip_max=1.,
                            loss_fn=loss_fn)

    total_pres = 0
    total_sum = 0
    success1_pres = 0
    success1_sum = 0
    success2_pres = 0
    success2_sum = 0
    failed_pres = 0
    failed_sum = 0

    for data, target in test_set:
        print("----------------------------------------------------------------------------------------")
        data, target = data.cuda(), target.cuda()

        adv_target1 = getTarget.get_best_target(model1, data, target, epsilon=eps, ord=2,
                                                loss_fn=loss_fn)
        adv_target2 = getTarget.get_best_target(model2, data, target, epsilon=eps, ord=2,
                                                loss_fn=loss_fn)

        # adv_target1 = target
        # adv_target2 = target

        # print("original target: ", target.item())
        model1_classification = model1(data).data.max(1)[1]
        model2_classification = model2(data).data.max(1)[1]
        # print("current classification by 1: {}, by 2: {}".format(model1_classification, model2_classification))
        # print("suggested best targets: 1 - {}, 2 - {}".format(adv_target1.item(), adv_target2.item()))

        if model1_classification.item() != model2_classification.item() \
                or model1_classification.item() != target.item() or model2_classification.item() != target.item():
            continue

        if adv_target1.item() != adv_target2.item():
            # print("****different best targets, choosing the first - {} ****".format(adv_target1))
            adv_target2 = adv_target1.clone()

        adv_img1 = attack1.perturb(data, adv_target1)
        adv_m1 = adv_img1 - data

        adv_img2 = attack2.perturb(data, adv_target2)
        adv_m2 = adv_img2 - data

        # print("angle between adversarial vectors: ", algebraTools.angle_between_tensor(adv_m1, adv_m2))
        # print("the L2 distance to the original image: 1: {}, 2: {}".format(torch.norm(adv_m1), torch.norm(adv_m2)))
        #
        # # ----------- projection on the local adversarial vectors -------------
        # print("\n++++++++ projection on the local adversarial vectors ++++++++++++")
        # linmatrix1 = find_matrix_around_image_batch(data, 0.01, model=model1)[0]
        # linmatrix2 = find_matrix_around_image_batch(data, 0.01, model=model2)[0]
        #
        # basis1 = algebraTools.create_orthonormal_basis(linmatrix1)
        # basis2 = algebraTools.create_orthonormal_basis(linmatrix2)
        #
        # # first lin space
        # print("\n$$ first linear space $$")
        # adv1_proj1 = algebraTools.project_on_basis(basis1, adv_m1.reshape(3 * 32 * 32))
        # adv2_proj1 = algebraTools.project_on_basis(basis1, adv_m2.reshape(3 * 32 * 32))
        #
        # print("angle between the 2 projections of adv on lin1:{}".format(
        #     algebraTools.angle_between_tensor(adv1_proj1, adv2_proj1)))
        #
        # # second linear space
        # print("\n$$ second linear space $$")
        # adv1_proj2 = algebraTools.project_on_basis(basis2, adv_m1.reshape(3 * 32 * 32))
        # adv2_proj2 = algebraTools.project_on_basis(basis2, adv_m2.reshape(3 * 32 * 32))
        #
        # print("angle between the 2 projection of adv on lin2:{}".format(
        #     algebraTools.angle_between_tensor(adv1_proj2, adv2_proj2)))
        #
        # # random linear space
        # print("\n$$ random linear space of dimension {} $$".format(linmatrix1.shape[0]))
        # rand_basis = get_random_manifold_basis(image_height=data.shape[-1], manifold_dim=linmatrix1.shape[0])
        # adv1_projrand = algebraTools.project_on_basis(rand_basis, adv_m1.reshape(3 * 32 * 32))
        # adv2_projrand = algebraTools.project_on_basis(rand_basis, adv_m2.reshape(3 * 32 * 32))
        # print("angle between the 2 projection of adv on lin2:{}".format(
        #     algebraTools.angle_between_tensor(adv1_projrand, adv2_projrand)))

        # ----------- projection on a low dimentional manifold -------------
        # print("\n++++++++ projection on the low dimensional image manifold ++++++++++++")
        encoder = cifar_encoder_manager.load_model(model_name="vgg16", exp_name="cifar-all-nopretrained")
        enc_basis = find_local_manifold_around_image(encoder=encoder, image=data)

        adv1_on = project_diff_on_basis(enc_basis, adv_m1)
        adv2_on = project_diff_on_basis(enc_basis, adv_m2)

        adv1_off = adv_m1 - adv1_on
        adv2_off = adv_m2 - adv2_on

        angle_on_manifold = algebraTools.angle_between_tensor(adv1_on, adv2_on)
        print("angle between the 2 projection of adv on manifold:{}".format(
            angle_on_manifold))

        print("angle between the 2 projection of adv off manifold:{}".format(
            algebraTools.angle_between_tensor(adv1_off, adv2_off)))

        # print("\n++ projection on a random low dimensional manifold of dimension {}++".format(enc_basis.shape[0]))
        rand_basis = get_random_manifold_basis(image_height=data.shape[-1], manifold_dim=enc_basis.shape[0])
        adv1_on_rand = project_diff_on_basis(rand_basis, adv_m1)
        adv2_on_rand = project_diff_on_basis(rand_basis, adv_m2)

        adv1_off_rand = adv_m1 - adv1_on_rand
        adv2_off_rand = adv_m2 - adv2_on_rand

        angle_on_random_manifold = algebraTools.angle_between_tensor(adv1_on_rand, adv2_on_rand)
        print("angle between the 2 projection of adv on random manifold:{}".format(
            angle_on_random_manifold))

        print("angle between the 2 projection of adv off random manifold:{}".format(
            algebraTools.angle_between_tensor(adv1_off_rand, adv2_off_rand)))

        angle_on_pres = 100 * (angle_on_manifold / angle_on_random_manifold)
        print("precent angle to random", angle_on_pres)
        total_pres += angle_on_pres
        total_sum += 1

        # print("cheating model 2", model2(adv_img1).data.max(1)[1], target, ~model2(adv_img1).data.max(1)[1].eq(target))
        if ~model2(adv_img1).data.max(1)[1].eq(target):
            success1_pres += angle_on_pres
            success1_sum += 1

        # print("cheating model 1", model1(adv_img2).data.max(1)[1], target, ~model1(adv_img2).data.max(1)[1].eq(target))
        if ~model1(adv_img2).data.max(1)[1].eq(target):
            success2_pres += angle_on_pres
            success2_sum += 1

        if model2(adv_img1).data.max(1)[1].eq(target) and model1(adv_img2).data.max(1)[1].eq(target):
            failed_pres += angle_on_pres
            failed_sum += 1

        print("total present (on manifold / on random manifold) {}, ({} / {})".format(total_pres/total_sum, total_pres, total_sum))
        if success1_sum > 0:
            print("start from model 1 successful attacks present (on manifold / on random manifold) {}, ({} / {})".format(success1_pres/success1_sum, success1_pres, success1_sum))
        if success2_sum > 0:
            print("start from model 2 successful attacks present (on manifold / on random manifold) {}, ({} / {})".format(success2_pres/success2_sum, success2_pres, success2_sum))
        if failed_sum > 0:
            print("failed attacks present (on manifold / on random manifold) {}, ({} / {})".format(failed_pres/failed_sum, failed_pres, failed_sum))


        # basicview.view_classification_changes_multirow([mang1, mang2], [data, data], [adv_img1, adv_img1])
        # basicview.view_classification_changes_multirow([mang1, mang2], [data, data], [adv_img2, adv_img2])


if __name__ == "__main__":
    exp1()
    # general_transferability()