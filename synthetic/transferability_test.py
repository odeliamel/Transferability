import torch

from synthetic.projections import project_diff_off_global_basis, project_diff_on_global_basis, \
    project_diff_on_local_on_global_basis, project_diff_on_global_off_local_basis


def test_projected_transferability(model1, projection_attack1, projection_function, target_model, data, target):
    adv = \
        projection_attack1.perturb(data, target, project=projection_function)
    print("projection function {}".format(projection_function.__name__))
    print("target: {}, model1 classification: {}, model 2 classification: {}".format(
        target.item(), model1(adv).sign().item(), target_model(adv).sign().item()))
    model2_classification = target_model(adv).sign()
    is_successful_attack = model1(adv).sign() != target
    is_transferred = model2_classification != target
    return float(is_transferred), float(is_successful_attack)


def test_projected_transferability_all(args, data_loader, model1, model2, projection_attack1, projection_attack2, projection_functions=None):
    from1_to2 = {}
    from2_to1 = {}

    successful_attacks1 = {}
    successful_attacks2 = {}

    def identity(x, y):
        return x

    if projection_functions is None:
        projection_functions = [identity, project_diff_off_global_basis, project_diff_on_global_basis,
                                project_diff_on_local_on_global_basis, project_diff_on_global_off_local_basis]

    for projection_function in projection_functions:
        from1_to2[projection_function.__name__] = torch.zeros(args.data_amount)
        from2_to1[projection_function.__name__] = torch.zeros(args.data_amount)
        successful_attacks1[projection_function.__name__] = torch.zeros(args.data_amount)
        successful_attacks2[projection_function.__name__] = torch.zeros(args.data_amount)

    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.cuda()
        target = target.cuda()
        # print(trained_model(data).data)
        for projection_function in projection_functions:
            from1_to2[projection_function.__name__][batch_idx], successful_attacks1[projection_function.__name__] = \
                test_projected_transferability(model1, projection_attack1, projection_function, model2, data, target)
            from2_to1[projection_function.__name__][batch_idx], successful_attacks2[projection_function.__name__] = \
                test_projected_transferability(model2, projection_attack2, projection_function, model1, data, target)

    for projection_function in projection_functions:
        print("out of {} successful attacks with projection {}, transferring adv from 1 to 2: {}".format(
            torch.sum(successful_attacks1[projection_function.__name__]),
            projection_function.__name__, torch.sum(from1_to2[projection_function.__name__])))
        print("out of {} successful attacks with projection {}, transferring adv from 2 to 1: {}".format(
            torch.sum(successful_attacks2[projection_function.__name__]),
            projection_function.__name__, torch.sum(from2_to1[projection_function.__name__])))
