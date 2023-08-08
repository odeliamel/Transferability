import sys

from advertorch.utils import batch_multiply
from algebra.projectionTools import get_projection_matrix

from highDimSynthetic.sphere.GetParams import get_args
import torch

from MNIST.PCAMNIST import load_PCA

global_basis_dim = 32
args = get_args(sys.argv[1:])
V = load_PCA(pca_path="../PCAs/pca32q10iters.pickle")["V"]
global_basis = V.T[:global_basis_dim, :].cuda()
global_projection_matrix = get_projection_matrix(global_basis)
print("global_basis_dim"+str(global_basis_dim))


def project_diff_on_global_basis(diff, data):
    batch_size = data.shape[0]
    diff = diff.detach().clone()
    # res[:, args.data_dim:] = 0
    global_projection_matrix_batch = global_projection_matrix.expand(batch_size, -1, -1)
    # print("{}, {}, {}".format(global_projection_matrix_batch.shape, local_proj_matrix_batch.shape, batch_v.shape))
    # joint_projection_matrix = torch.bmm(local_proj_matrix_batch, global_projection_matrix_batch)
    flat_batch_v = diff.reshape(batch_size, 784, 1)
    # print(joint_projection_matrix.shape)
    # res = torch.bmm(joint_projection_matrix, flat_batch_v)
    res = torch.bmm(global_projection_matrix_batch, flat_batch_v)
    res = res.reshape_as(diff)
    return res


def project_diff_off_global_basis(diff, data):
    res = diff.clone()
    on = project_diff_on_global_basis(diff, data)
    return res - on

