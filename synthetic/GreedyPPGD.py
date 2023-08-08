import numpy as np
import torch
import torch.nn as nn

from advertorch.utils import clamp
from advertorch.utils import normalize_by_pnorm
from advertorch.utils import clamp_by_pnorm
from advertorch.utils import is_float_or_torch_tensor
from advertorch.utils import batch_multiply
from advertorch.utils import batch_clamp
from advertorch.utils import replicate_input
import attacks.EBT.getTarget as EBT
from attacks.PGD import _get_predicted_label
from attacks.abstract_attack import AbstractAttack
from attacks.greedy_step import get_smallest_step

from advertorch.attacks.base import Attack
from advertorch.attacks.base import LabelMixin
from advertorch.attacks.utils import rand_init_delta
from advertorch.attacks.utils import is_successful


def perturb_iterative(xvar, yvar, predict, project, nb_iter, eps, eps_iter, loss_fn,
                      delta_init=None, minimize=True, ord=np.inf,
                      clip_min=0.0, clip_max=1.0):
    # print("in perturb iterative")
    if delta_init is not None:
        delta = delta_init
    else:
        delta = torch.zeros_like(xvar)

    predict.zero_grad()
    delta.requires_grad_()

    for ii in range(nb_iter):
        predict.zero_grad()
        outputs = predict(xvar + delta)
        preds = outputs.sign().squeeze(1)
        # print(yvar.shape, outputs.shape, preds.shape)
        # if project.__name__ == "project_diff_on_global_off_local_basis":
            # print(ii, preds, yvar)

        to_improve = ~is_successful(preds, yvar, minimize)
        # print(preds, yvar, minimize)
        # print(torch.norm(delta.data, p=2, dim=-1))
        # print(ii, "to improve", to_improve, project.__name__)
        if to_improve.sum() == 0:
            # print("to improve", to_improve, project.__name__)
            break

        loss = loss_fn(outputs[to_improve], yvar[to_improve])
        if minimize:
            loss = -loss
        loss.backward()

        if ord == np.inf:
            grad = delta.grad.data
            grad_sign = grad[to_improve].sign()
            delta.data[to_improve] = delta.data[to_improve] + batch_multiply(eps_iter, grad_sign)
            delta.data[to_improve] = batch_clamp(eps, delta.data[to_improve])
            delta.data[to_improve] = clamp(xvar.data[to_improve] + delta.data[to_improve], clip_min, clip_max
                               ) - xvar.data[to_improve]

        elif ord == 2:
            grad = delta.grad.data
            # grad_p = project(grad)
            grad = project(grad[to_improve], xvar.data[to_improve])
            # grad = grad[to_improve]
            # print("1", torch.norm(grad, p=2))
            grad = normalize_by_pnorm(grad, small_constant=1e-19)
            # print("2", torch.norm(grad, p=2))
            delta.data[to_improve] = delta.data[to_improve] + batch_multiply(eps_iter, grad)
            # delta.data[to_improve] = project(xvar.data[to_improve] + delta.data[to_improve]) - xvar.data[to_improve]
            # print("3", torch.norm(delta.data[to_improve], p=2))
            delta.data[to_improve] = clamp(xvar.data[to_improve] + delta.data[to_improve], clip_min, clip_max
                               ) - xvar.data[to_improve]
            if eps is not None:
                delta.data[to_improve] = clamp_by_pnorm(delta.data[to_improve], ord, eps)
            # print("4", torch.norm(delta.data[to_improve]))
            delta.data[to_improve] = project(delta.data[to_improve], xvar.data[to_improve])
            # print("5", torch.norm(delta.data[to_improve]))

        else:
            error = "Only ord = inf and ord = 2 have been implemented"
            raise NotImplementedError(error)

        delta.grad.data.zero_()
        predict.zero_grad()

    x_adv = clamp(xvar + delta.data, clip_min, clip_max)
    return x_adv


class GreedyPPGDAttack(Attack, LabelMixin):

    def __init__(self, predict, loss_fn=None, eps=8/255, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            ord=2, targeted=True):

        super(GreedyPPGDAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.ord = ord
        self.targeted = targeted
        self.loss_fn = loss_fn
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

        assert is_float_or_torch_tensor(self.eps_iter)
        assert is_float_or_torch_tensor(self.eps)

    def perturb(self, x, y=None, project=None):
        # print("in perturb")
        x, y = self._verify_and_process_inputs(x, y)

        if project is None:
            project = lambda v, u: v

        rval_0 = perturb_iterative(
            x, y, self.predict, project, nb_iter=self.nb_iter,
            eps=self.eps, eps_iter=self.eps_iter,
            loss_fn=self.loss_fn, minimize=self.targeted,
            ord=self.ord, clip_min=self.clip_min,
            clip_max=self.clip_max, delta_init=None)

        res = rval_0.data.detach().clone()

        # print("final", self.predict(res).data.max(1)[1])
        # return self.update_adv(x, y, res, 0.)
        return res
        # return clamp(res, self.clip_min, self.clip_max)

    def update_adv(self, x, y, adv, min_probability):
        direction = (adv-x).clone()
        delta = get_smallest_step(self.predict, x, direction, y, self.eps, targeted=self.targeted, min_probability=min_probability)

        if self.eps is not None:
            delta = clamp_by_pnorm(delta, 2, self.eps)

        return clamp(x + delta, self.clip_min, self.clip_max)

    def get_distance_to_boundary(self, x, y=None, project=None):
        adv = self.perturb(x, y, project)
        # print((adv-x).shape)
        norm = torch.norm(adv-x, p=2, dim=-1)
        # print(x.shape, norm.shape)
        # print("norm", norm)
        return norm, is_successful(self.predict(adv).sign()[0], y, self.targeted)

#
# import torch
# import torch.nn.functional as F
# from advertorch.utils import normalize_by_pnorm
# from advertorch.utils import batch_multiply
# from advertorch.attacks.utils import is_successful

#
# def get_smallest_step(predict, x, direction, target, epsilon, targeted, fraction=100, min_probability=0, t=2):
#     x = x.detach()
#     max_norm = torch.where(is_successful(predict(x+direction).data.max(1)[1], target, targeted),
#                            torch.norm(direction, dim=[1, 2, 3], p=2),
#                            epsilon*torch.ones(x.shape[0], device='cuda'))
#
#     normalized_direction = normalize_by_pnorm(direction.detach(), p=2, small_constant=1e-19)
#     eps_direction = batch_multiply(max_norm, normalized_direction)
#     # print(torch.norm(eps_direction, dim=[1, 2, 3], p=2))
#     res = torch.zeros_like(eps_direction, device='cuda')
#     rang = torch.arange(0, fraction + 1, 1, dtype=torch.float, device='cuda').reshape(fraction + 1, 1, 1, 1) / fraction
#     # res = torch.zeros_like(eps_direction)
#     with torch.no_grad():
#         for k in range(x.shape[0]):
#             start_image = x[k].detach().clone()
#             direct = eps_direction[k].detach().clone()
#             for iter in range(t):
#                 # print(torch.norm(direct, p=2))
#                 # print("eps direction: ", torch.norm(eps_direction[k], p=2))
#                 allintervals = start_image.repeat(fraction+1, 1, 1, 1) + direct.repeat(fraction+1, 1, 1, 1) * rang
#                 # print(torch.norm(start_image - allintervals[-1], p=2))
#                 output = predict(allintervals).detach()
#                 preds = torch.argmax(output, dim=1)
#                 # print("all interval preds", preds)
#                 # probs = F.softmax(output, dim=1)
#                 success = is_successful(preds, target[k], targeted)
#                 # print(success)
#                 # prob_success = torch.max(probs, dim=1)[0] >= min_probability
#
#                 # firstone = ((success & prob_success) == 0).sum()
#                 firstone = ((success) == 0).sum()
#                 if firstone == rang.shape[0]:
#                     direct = rang[fraction].item() * direct
#                 else:
#                     direct = rang[firstone.item()].item() * direct
#
#             res[k] = direct
#     # if t == 0:
#     return res
#     # else:
    #     return get_smallest_step(predict, x, res, target, epsilon, targeted, fraction=100, min_probability=0, t=t-1)


