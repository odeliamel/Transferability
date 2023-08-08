import sys, os

from advertorch.attacks import PGDAttack
from algebra.algebraTools import angle_between_tensor
from attacks.PGD import PGDAttack as PGDAttack_nopredict
from attacks.auto_pgd import APGD
from views import basicview

from synthetic import gradients_test
from synthetic import robustness_test
from synthetic.GreedyPPGD import GreedyPPGDAttack
from synthetic.PPGD import PPGDAttack
from synthetic.transferability_test import test_projected_transferability_all

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from threadpoolctl import threadpool_limits

_thread_limit = threadpool_limits(limits=8)

from synthetic.GetParams import get_args
from synthetic.CreateData import create_data
from synthetic.CreateModel import create_model
from synthetic.utils import *
from synthetic.AdvPert import *
from synthetic.projections import project_diff_on_global_basis, project_diff_off_global_basis, \
    project_diff_on_global_off_local_basis, project_diff_on_local_on_global_basis

from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler
import torch


def train(args, dataset, model):
    batch_size = get_batch_size(args)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    parameters = list(model.parameters())
    optimizer = SGD(parameters, lr=args.train_lr, weight_decay=0.)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10000, gamma=1.1)

    for epoch in range(1, args.train_iterations + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.cuda()
            target = target.cuda()
            optimizer.zero_grad()
            # batch = batch.to(device)
            output = model(data.float())
            loss = calc_loss(args, (data, target), output)
            loss.backward()
            # print(torch.norm(model.layers[0].weight.grad.data[:, args.data_dim:]))
            optimizer.step()
            scheduler.step()

        if epoch % args.train_evaluate_rate == 0:
            acc = evaluate(args, model, dataloader, epoch)
            print(acc)
            if acc == 1:
                return model


    return model


def adversarial_train(args, dataset, model):
    batch_size = get_batch_size(args)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    for layer in model.layers:
        layer.weight = torch.nn.Parameter(layer.weight / 50)
    parameters = list(model.parameters())
    optimizer = SGD(parameters, lr=args.train_lr, weight_decay=0.)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10000, gamma=1.1)
    adv_train_attack = GreedyPPGDAttack(predict=model, eps=6., nb_iter=200, eps_iter=.1, ord=2,
                                        targeted=False,
                                        loss_fn=lambda x, y: calc_loss(args, [0, x], y),
                                        clip_min=-100000, clip_max=100000, rand_init=False)
    for epoch in range(1, args.train_iterations + 1):
        # model.train()
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.cuda()
            target = target.cuda()
            optimizer.zero_grad()
            model.eval()
            adv = torch.zeros_like(data)
            # for i in range(data.shape[0]):
            #     adv[i] = adv_train_attack.perturb(data[i:i+1], target[i:i+1], project=project_diff_on_global_basis)
            # adv_diff = adv - data
            # adv_projected = data + project_diff_on_global_basis(adv_diff, data)
            model.train()
            # batch = batch.to(device)
            # output = model(adv)
            # loss = calc_loss(args, (adv, target), output)
            output = model(data.float())
            loss = calc_loss(args, (data, target), output)
            loss.backward()
            optimizer.step()
            scheduler.step()

        if epoch % args.train_evaluate_rate == 0:
            acc = evaluate(args, model, dataloader, epoch)
            if acc == 1:
                return model



    return model


def main():
    args = get_args(sys.argv[1:])

    torch.manual_seed(args.seed)
    device = torch.device("cuda:{}".format(args.gpu) if args.cuda else "cpu")

    model1 = create_model(args)
    model2 = create_model(args)
    # model_robust = create_model(args)
    dataset = create_data(args)

    for i in range(len(model1.layers)):
        print(model1.layers[i])
        # model2.layers[i].weight = torch.nn.Parameter(model1.layers[i].weight.clone())
        print("weight vector distance", torch.norm(model2.layers[i].weight - model1.layers[i].weight))
        # model2.layers[i].bias = torch.nn.Parameter(model1.layers[i].bias.clone())

    trained_model_unit1 = train(args, dataset, model1)
    # trained_model_unit2 = train(args, dataset, model2)
    trained_model_unit2 = adversarial_train(args, dataset, model2)

    # attack = PGDAttack(predict=model_robust, eps=15., nb_iter=1, eps_iter=15., ord=2, targeted=False,
    #                     loss_fn=lambda x, y: calc_loss(args, [0, x], y),
    #                     clip_min=-100000, clip_max=100000, rand_init=False)
    # trained_model_robust = adversarial_train(args, dataset, model_robust, attack)
    # trained_model_unit = model
    trained_model1 = trained_model_unit1.eval().cuda()
    trained_model2 = trained_model_unit2.eval().cuda()
    # trained_model2 = trained_model_robust.eval().cuda()
    # trained_model1 = model1.eval().cuda()
    # trained_model2 = model2.eval().cuda()

    # trained_model = lambda x: torch.tensor([[-trained_model_unit(x)-1, 1-trained_model_unit(x)]]).cuda()

    attack1 = PGDAttack_nopredict(eps=15., nb_iter=1, eps_iter=15., ord=2, targeted=False,
                                  loss_fn=lambda output, target: calc_adv_loss(args, [0, target], output),
                                  clip_min=-100000, clip_max=100000, rand_init=False)

    greedy_attack1 = GreedyPPGDAttack(predict=trained_model1, eps=50., nb_iter=400, eps_iter=.5, ord=2, targeted=False,
                                      loss_fn=lambda output, target: calc_adv_loss(args, [0, target], output),
                                      clip_min=-100000, clip_max=100000, rand_init=False)

    greedy_attack2 = GreedyPPGDAttack(predict=trained_model2, eps=50., nb_iter=400, eps_iter=.5, ord=2, targeted=False,
                                      loss_fn=lambda output, target: calc_adv_loss(args, [0, target], output),
                                      clip_min=-100000, clip_max=100000, rand_init=False)

    projection_attack1 = PPGDAttack(predict=trained_model1, eps=15., nb_iter=200, eps_iter=.5, ord=2, targeted=False,
                                      loss_fn=lambda output, target: calc_adv_loss(args, [0, target], output),
                                      clip_min=-100000, clip_max=100000, rand_init=False)

    projection_attack2 = PPGDAttack(predict=trained_model2, eps=15., nb_iter=200, eps_iter=.5, ord=2, targeted=False,
                                      loss_fn=lambda output, target: calc_adv_loss(args, [0, target], output),
                                      clip_min=-100000, clip_max=100000, rand_init=False)

    # for i in range(len(model1.layers)):
    #     print("2weight vector distance", model1.layers[i], torch.norm(model2.layers[i].weight -
    #                                                                   model1.layers[i].weight))
    #     print("3weight vector distance", model1.layers[i], torch.norm(model2.layers[i].weight[:, args.data_dim:] -
    #                                                                   model1.layers[i].weight[:, args.data_dim:]))

    # gradients_test.test_angles(args, trained_model1, trained_model2, dataloader, attack1)
    robustness_test.test_robustness(args, dataset, greedy_attack1, greedy_attack2)
    # test_projected_transferability_all(args, dataloader, model1, model2, projection_attack1, projection_attack2)

    # TODO: see if the margin between classes is closing like linear function with cosine ratio!


if __name__ == '__main__':
    main()
