import torch
from algebra.algebraTools import angle_between_tensor


def get_projected_norms_and_angles(v1, v2, projection_function=None):
    if projection_function is None:
        projection_function = lambda v: v

    v1_proj = projection_function(v1)
    v2_proj = projection_function(v2)
    v1_proj_norm = torch.norm(v1_proj, p=2)
    v2_proj_norm = torch.norm(v2_proj, p=2)

    angle = angle_between_tensor(v1_proj, v2_proj)

    return v1_proj_norm, v2_proj_norm, angle

