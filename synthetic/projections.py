import sys

from advertorch.utils import batch_multiply

from synthetic.GetParams import get_args
import torch

args = get_args(sys.argv[1:])


def project_diff_on_global_basis(diff, data):
    res = diff.clone()
    res[:, args.data_dim:] = 0
    return res


def project_diff_off_global_basis(diff, data):
    res = diff.clone()
    res[:, 0:args.data_dim] = 0
    return res


def project_diff_on_local_on_global_basis(diff, data):
    x_data_dim = diff.clone()
    x_data_dim[:, args.data_dim:] = 0
    x_sphere = args.data_uniform_norm * batch_multiply(1 / torch.norm(x_data_dim+data, p=2, dim=-1), (x_data_dim+data))
    res = x_sphere - data
    return res


def project_diff_on_global_off_local_basis(diff, data):
    x_data_dim = diff.clone()
    x_data_dim[:, args.data_dim:] = 0
    x_sphere = args.data_uniform_norm * batch_multiply(1 / torch.norm(x_data_dim+data, p=2, dim=-1), (x_data_dim+data))
    res = x_data_dim - (x_sphere - data)
    # print("vectors norms: ", torch.norm(x_data_dim), torch.norm(x_sphere-data), torch.norm(res))
    return res
