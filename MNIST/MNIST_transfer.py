import torch
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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



class Normalize(nn.Module):
    def __init__(self, mean, std, ndim=4, channels_axis=1, dtype=torch.float32):
        super(Normalize, self).__init__()
        shape = tuple(-1 if i == channels_axis else 1 for i in range(ndim))
        mean = torch.tensor(mean, dtype=dtype).reshape(shape)
        std = torch.tensor(std, dtype=dtype).reshape(shape)
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

    def forward(self, x):
        return (x.cuda() - self.mean.cuda()) / self.std.cuda()


class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.model = model.mnist(input_dims=784, n_hiddens=[256, 256], n_class=10, pretrained=True, src="mnist")
        # normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # self.model = nn.Sequential(normalize, self.model)
        self.model = torch.nn.DataParallel(self.model)

        self.model = self.model.cuda()
        self.model = self.model.eval()
        self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    def forward(self, input):
        return self.model(input)

    def runModelOnImage(self, image, logging=True, withprobability=False):
        image = image.type(torch.FloatTensor).cuda()
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        output = self.model(image)[0]
        prob = output.softmax(0)
        probability, pred = prob.topk(k=1, dim=0)
        # print(pred)
        # print(probability)
        predclass = self.classes[pred.item()]

        if logging:
            resstr = "\npred: " + str(
                pred) + "\nclass: " + predclass + ", with probability: " + str(probability.item())
            print("Running cifar on image: ", resstr)

        if withprobability:
            return output, predclass, probability.item()

        return output, predclass


class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.model = model.mnist(input_dims=784, n_hiddens=[128, 128], n_class=10, pretrained=True, src="mnist_dup")
        # normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # self.model = nn.Sequential(normalize, self.model)
        self.model = torch.nn.DataParallel(self.model)

        self.model = self.model.cuda()
        self.model = self.model.eval()
        self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    def forward(self, input):
        return self.model(input)

    def runModelOnImage(self, image, logging=True, withprobability=False):
        image = image.type(torch.FloatTensor).cuda()
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        output = self.model(image)[0]
        prob = output.softmax(0)
        probability, pred = prob.topk(k=1, dim=0)
        # print(pred)
        # print(probability)
        predclass = self.classes[pred.item()]

        if logging:
            resstr = "\npred: " + str(
                pred) + "\nclass: " + predclass + ", with probability: " + str(probability.item())
            print("Running cifar on image: ", resstr)

        if withprobability:
            return output, predclass, probability.item()

        return output, predclass


# def exp3():
#     model1 = Model1()
#     model2 = Model2()
#
#     model1, model2 = model1.eval(), model2.eval()
#     model1, model2 = model1.cuda(), model2.cuda()
#
#     # encoder = ImagenetEncoder()
#     test_set = dataset.IMAGENETdata.getTestSetIterator(batch_size=1)
#
#     eps = 16 / 255
#     eps_iter = 2 * 16 / 255 / 50
#     # eps = 1.
#     # eps_iter = 2 * eps / 50
#     loss_fn = imagenet_margin_loss
#
#     # attack1 = PGD.PGDAttack(predict=model1.model, eps=eps, nb_iter=50, eps_iter=eps_iter,
#     #                         rand_init=True, ord=float("inf"), targeted=True, clip_min=0., clip_max=1.,
#     #                         loss_fn=loss_fn)
#
#     attack1 = OnManifoldPGDAttack(encoder=encoder, predict=model1.model, eps=eps, nb_iter=50, eps_iter=eps_iter,
#                             rand_init=True, ord=float("inf"), targeted=True, clip_min=0., clip_max=1.,
#                             loss_fn=loss_fn)
#
#     for data, target in test_set:
#         print("------------------------------")
#         data, target = data.cuda(), target.cuda()
#         adv_target1 = getTarget.get_best_target(model1.model, data, target, epsilon=eps, ord=2,
#                                                 loss_fn=loss_fn)
#         print("adv target", adv_target1)
#         adv_m1 = attack1.perturb(data, adv_target1)
#         basis = projectionTools.find_local_manifold_around_image(encoder, data)
#         adv_on = projectionTools.project_diff_on_basis(basis, adv_m1 - data)
#
#         print("adv norm {} adv on norm {}". format(torch.norm(adv_m1-data, p=2), torch.norm(adv_on, p=2)))
#         basicview.view_classification_changes_multirow(model1, [data] * 3,
#                                                        [adv_m1, data + adv_on, data + (adv_m1 - data - adv_on)])
#         basicview.view_classification_changes_multirow(model2, [data] * 3,
#                                                        [adv_m1, data + adv_on, data + (adv_m1 - data - adv_on)])


def exp2():
    model1 = Model1()
    model2 = Model2()

    model1, model2 = model1.eval(), model2.eval()
    model1, model2 = model1.cuda(), model2.cuda()

    # encoder = ImagenetEncoder()
    encoder = MnistEncoder()
    # test_set = dataset.IMAGENETdata.getTestSetIterator(batch_size=100)
    test_set = dataset.MNISTdata.getTestSetIterator(batch_size=1)
    # test_set = dataset_wraper.get_all_image_from_class(test_set, 1, batch_size=1)
    # test_set = save_and_load.load_images_from_folder(folder="imagenet", name="fishnshark224-val",
    #                                                  batch_size=1)

    # eps = 16 / 255
    eps = 2.
    eps_iter = eps
    # loss_fn = imagenet_margin_loss
    loss_fn = margin_loss
    # attack1 = PGD.PGDAttack(predict=model1.model, eps=eps, nb_iter=50, eps_iter=eps_iter,
    #                         rand_init=False, ord=2, targeted=False, clip_min=0., clip_max=1.,
    #                         loss_fn=loss_fn)

    attack1 = OnManifoldPGDAttack(encoder=encoder, predict=model1.model, eps=eps, nb_iter=50, eps_iter=eps_iter,
                                  rand_init=False, ord=2, targeted=True, clip_min=0., clip_max=1.,
                                  loss_fn=loss_fn)

    target_fn = lambda data, target: getTarget.get_best_target(model1.model, data, target, epsilon=eps, ord=2,
                                            loss_fn=loss_fn)

    # target_fn = lambda data, target: target

    check_transferability(model1, model2, attack1, test_set, target_fn)

def exp1():
    model1 = Model1()
    model2 = Model2()

    model1, model2 = model1.eval(), model2.eval()
    model1, model2 = model1.cuda(), model2.cuda()

    encoder = MnistEncoder()

    on_manifold_angles = torch.tensor([]).cuda()
    off_manifold_angles = torch.tensor([]).cuda()

    # eps = 16/255
    # eps_iter = 2 * eps / 50
    eps = 2.
    eps_iter = 0.1
    loss_fn = margin_loss

    test_set = dataset.MNIST.getTestSetIterator(batch_size=1)

    attack1 = PGD.PGDAttack(predict=model1.model, eps=eps, nb_iter=50, eps_iter=eps_iter,
                           rand_init=False, ord=2, targeted=False, clip_min=0., clip_max=1.,
                           loss_fn=loss_fn)

    attack2 = PGD.PGDAttack(predict=model2.model, eps=eps, nb_iter=50, eps_iter=eps_iter,
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
        model1_classification = model1.model(data).data.max(1)[1]
        model2_classification = model2.model(data).data.max(1)[1]
        print("current classification by 1: {}, by 2: {}".format(model1_classification, model2_classification))
        print("suggested best targets: 1 - {}, 2 - {}".format(adv_target1.item(), adv_target2.item()))

        if adv_target1.item() != adv_target2.item() or model1_classification.item() != model2_classification.item()\
                or model1_classification.item() != target.item() or model2_classification.item() != target.item():
            continue

        adv_m1 = attack1.perturb(data, adv_target1)
        adv_m2 = attack2.perturb(data, adv_target2)

        print("angle between adversarial vectors : ", algebraTools.angle_between_tensor(adv_m1 - data, adv_m2 - data))
        # print("angle between adversarial vectors - the diff norm : ", torch.norm((adv_m1 - adv_m2), p=2))

        encoded_image = encoder.encode_and_decode(data)

        adv1_diff = adv_m1 - data
        adv2_diff = adv_m2 - data
        #
        # print("AE diff norm: ", torch.norm((encoded_image - data), p=2))
        # print("angle between adversarial vectors - the diff from adv to AE image", algebraTools.angle_between_tensor(adv1_diff_to_ae, adv2_diff_to_ae))

        basis = projectionTools.find_local_manifold_around_image(encoder, data)

        # data_proj = encoded_image + projectionTools.project_diff_on_basis(basis, data - encoded_image)

        # print("projection diff norm:", torch.norm(data_proj - data))
        # adv1_proj = encoded_image + projectionTools.project_diff_on_basis(basis, adv1_diff_to_ae)
        # adv2_proj = encoded_image + projectionTools.project_diff_on_basis(basis, adv2_diff_to_ae)

        adv_m1_proj_on = data + projectionTools.project_diff_on_basis(basis, adv1_diff)
        adv_m2_proj_on = data + projectionTools.project_diff_on_basis(basis, adv2_diff)

        print("on manifold distances:")
        print("on manifold adversarial distance 1:", torch.norm(adv_m1_proj_on - data, p=2))
        print("on manifold adversarial distance 2:", torch.norm(adv_m2_proj_on - data, p=2))
        on_manifold_angle = algebraTools.angle_between_tensor(adv_m1_proj_on - data, adv_m2_proj_on - data)
        print("on manifold angle between adversarial vectors:", on_manifold_angle)
        on_manifold_angles = torch.cat((on_manifold_angles, on_manifold_angle.unsqueeze(0)), dim=0)

        # data_off_manifold = data - data_proj
        adv_m1_off_manifold = data + (adv1_diff - projectionTools.project_diff_on_basis(basis, adv1_diff))
        adv_m2_off_manifold = data + (adv2_diff - projectionTools.project_diff_on_basis(basis, adv2_diff))

        print("off manifold adversarial distance 1:", torch.norm(adv_m1_off_manifold - data, p=2))
        print("off manifold adversarial distance 2:", torch.norm(adv_m2_off_manifold - data, p=2))
        off_manifold_angle = algebraTools.angle_between_tensor(adv_m1_off_manifold - data, adv_m2_off_manifold - data)
        print("off manifold angle between adversarial vectors:", off_manifold_angle)
        off_manifold_angles = torch.cat((off_manifold_angles, off_manifold_angle.unsqueeze(0)), dim=0)

        #TODO: see if the margin between classes is closing like linear function with cosine ratio!

        basicview.view_classification_changes_multirow([model1.model]*4, [data]*4, [adv_m1, adv_m1_off_manifold, adv_m2, adv_m2_off_manifold],
                                                       txt=["model 1, adv 1", "model 1 adv 1 proj off","model 1 adv 2", "model 1 adv2 proj off"])
        basicview.view_classification_changes_multirow([model2.model]*4, [data]*4, [adv_m1, adv_m1_off_manifold, adv_m2, adv_m2_off_manifold],
                                                       txt=["model 2, adv 1", "model 2 adv 1 proj off","model 2 adv 2", "model 2 adv2 proj off"])

        print("on manifold angle mean:", on_manifold_angles.mean())
        print("off manifold angle mean: ", off_manifold_angles.mean())



if __name__ == '__main__':
    exp1()
    # exp2()
    # exp3()