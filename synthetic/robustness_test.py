import torch
from torch.utils.data import DataLoader

from synthetic.projections import project_diff_off_global_basis, project_diff_on_global_basis, \
    project_diff_on_local_on_global_basis, project_diff_on_global_off_local_basis


def test_robustness(args, dataset, greedy_attack1, greedy_attack2):
    avg_distances1 = {}
    successful_attacks1 = {}
    avg_distances2 = {}
    successful_attacks2 = {}

    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size)

    if args.data_amount > 200:
        sample_size = 200
    else:
        sample_size = args.data_amount


    def identity(x, y):
        return x

    projection_functions = [identity, project_diff_off_global_basis, project_diff_on_global_basis,
                            project_diff_on_local_on_global_basis, project_diff_on_global_off_local_basis]

    for projection_function in projection_functions:
        avg_distances1[projection_function.__name__] = torch.zeros(sample_size)
        avg_distances2[projection_function.__name__] = torch.zeros(sample_size)
        successful_attacks1[projection_function.__name__] = torch.zeros(sample_size)
        successful_attacks2[projection_function.__name__] = torch.zeros(sample_size)

    for batch_idx, (data, target) in enumerate(dataloader):
        if (batch_idx) * batch_size >= sample_size:
            break
        data = data.cuda()
        target = target.cuda()
        # print(trained_model(data).data)
        # print(batch_idx, batch_size, sample_size)
        for projection_function in projection_functions:
            avg_distances1[projection_function.__name__][batch_idx * batch_size:(batch_idx+1) * batch_size], \
            successful_attacks1[projection_function.__name__][batch_idx * batch_size:(batch_idx+1) * batch_size] = \
                greedy_attack1.get_distance_to_boundary(data, target, project=projection_function)
            avg_distances2[projection_function.__name__][batch_idx * batch_size:(batch_idx+1) * batch_size], \
            successful_attacks2[projection_function.__name__][batch_idx * batch_size:(batch_idx+1) * batch_size] = \
                greedy_attack2.get_distance_to_boundary(data, target, project=projection_function)



    for projection_function in projection_functions:
        print(avg_distances1[projection_function.__name__].shape)
        print("out of {} successful attacks with projection {}, mean adversarial distance of model1: {}".format(
            torch.sum(successful_attacks1[projection_function.__name__]), projection_function.__name__, torch.mean(
                avg_distances1[projection_function.__name__])))
        print("out of {} successful attacks with projection {}, mean adversarial distance of model2: {}".format(
            torch.sum(successful_attacks2[projection_function.__name__]), projection_function.__name__, torch.mean(
                avg_distances2[projection_function.__name__])))

