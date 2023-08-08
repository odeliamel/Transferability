import torch
from algebra import algebraTools
from algebra.projectionTools import project_diff_on_basis, project_diff_off_basis


def get_relative_norm(total_vector, part_vector):
    total_norm = torch.norm(total_vector, p=2)
    part_norm = torch.norm(part_vector, p=2)

    return (part_norm / total_norm) * 100

def get_relative_effect(image, model, total_vector, part_vector, target_class, adv_class):
    # print(image.shape, total_vector.shape, part_vector.shape)
    with torch.no_grad():
        org_logits = model(image).data[0]
        part_logits = model(image+part_vector).data[0]
        total_logits = model(image+total_vector).data[0]
    # print(org_logits, part_logits, total_logits)
    org_diff = org_logits[target_class] - org_logits[adv_class]
    part_diff = part_logits[target_class] - part_logits[adv_class]
    total_diff = total_logits[target_class] - total_logits[adv_class]
    # print(org_diff, part_diff, total_diff)

    res = 100 * (part_diff - org_diff) / (total_diff - org_diff)
    return res.item(), (part_diff - org_diff).item(), (total_diff - org_diff).item()

def get_vector_effect_on_decision(image, model, added_vector, target_class, adv_class):
    print(image.shape, added_vector.shape)
    with torch.no_grad():
        org_logits = model(image)
        # new_logits = model(image+added_vector)

    org_diff = org_logits[target_class] - org_logits[adv_class]
    # new_diff = new_logits[target_class] - new_logits[adv_class]

    # return torch.norm(new_diff - org_diff, p=2)


def project_on_off_manifold(manifold_basis, rand_basis, vecor1, vector2):
    print("\n vectors projected on and off manifold")
    vector1_proj = project_diff_on_basis(manifold_basis, vecor1)
    vector2_proj = project_diff_on_basis(manifold_basis, vector2)

    vector1_offproj = project_diff_off_basis(manifold_basis, vecor1)
    vector2_offproj = project_diff_off_basis(manifold_basis, vector2)
    print("on manifold projected vector's norm: adv1: {}, adv2: {}".format(torch.norm(vector1_proj, p=2),
                                                                           torch.norm(vector2_proj, p=2)))
    print("angle between the adv. vectors projected on-manifold: {}".format(
        algebraTools.angle_between_tensor(vector1_proj, vector2_proj)))
    print("off manifold projected vector's norm: adv1: {}, adv2: {}".format(torch.norm(vector1_offproj, p=2),
                                                                            torch.norm(vector2_offproj, p=2)))
    print("angle between the adv. vectors projected off-manifold: {}".format(
        algebraTools.angle_between_tensor(vector1_offproj, vector2_offproj)))

    print("\n vectors projected on and off random manifold")
    vector1_proj = project_diff_on_basis(rand_basis, vecor1)
    vector2_proj = project_diff_on_basis(rand_basis, vector2)

    vector1_offproj = project_diff_off_basis(rand_basis, vecor1)
    vector2_offproj = project_diff_off_basis(rand_basis, vector2)

    print("on manifold projected vector's norm: adv1: {}, adv2: {}".format(torch.norm(vector1_proj, p=2),
                                                                           torch.norm(vector2_proj,
                                                                                      p=2)))
    print("angle between the adv. vectors projected on-manifold: {}".format(
        algebraTools.angle_between_tensor(vector1_proj, vector2_proj)))
    print("off manifold projected vector's norm: adv1: {}, adv2: {}".format(torch.norm(vector1_offproj, p=2),
                                                                            torch.norm(vector2_offproj, p=2)))
    print("angle between the adv. vectors projected off-manifold: {}".format(
        algebraTools.angle_between_tensor(vector1_offproj, vector2_offproj)))

