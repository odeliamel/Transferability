import sys

from Autoencoders.mnist_encoder.autoencoder import MnistEncoder
from advertorch.utils import batch_multiply
from algebra.projectionTools import get_projection_matrix, project_diff_on_manifold_around_image, \
    find_local_manifold_around_image

from highDimSynthetic.sphere.GetParams import get_args
import torch

from MNIST.PCAMNIST import load_PCA

global_basis_dim = 32
args = get_args(sys.argv[1:])
V = load_PCA(pca_path="../PCAs/pca32q10iters.pickle")["V"]
global_basis = V.T[:global_basis_dim, :].cuda()
global_projection_matrix = get_projection_matrix(global_basis)
print("global_basis_dim"+str(global_basis_dim))

encoder = MnistEncoder()


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


def project_diff_on_local_on_global_basis(diff, data):
    on_local = _project_diff_on_local_manifold(diff.clone(), data)
    on_local_on_global = project_diff_on_global_basis(on_local.clone(), data)
    # print("diff_norm", torch.norm(diff, p=2))
    # print("diff_norm on_global", torch.norm(on_global, p=2))
    # print("diff_norm on_local_on_global", torch.norm(on_local_on_global, p=2))
    # print("diff on on ", torch.norm(on_global - on_local_on_global, p=2))
    return on_local_on_global


def project_diff_on_global_off_local_basis(diff, data):
    on_local = _project_diff_on_local_manifold(diff.clone(), data)
    off_local = diff - on_local
    off_local_on_global = project_diff_on_global_basis(off_local, data)
    # print("diff off on", torch.norm(on_global-off_local_on_global, p=2))
    return off_local_on_global


def _project_diff_on_local_manifold(diff, data):
    basis = find_local_manifold_around_image(encoder, data)
    # print("basis", basis.shape)
    one = torch.matmul(basis, diff.view(diff.shape[0], -1).unsqueeze(-1))
    # print("one", one.shape)
    proj = (one * basis).sum(dim=1)
    # print("proj", proj.shape)
    return proj.reshape_as(diff)
