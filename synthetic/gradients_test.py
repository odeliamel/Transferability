import torch
from algebra.algebraTools import angle_between_tensor

from synthetic.projections import project_diff_on_global_off_local_basis, project_diff_on_local_on_global_basis, \
    project_diff_off_global_basis, project_diff_on_global_basis
from synthetic.test_utils import get_projected_norms_and_angles


def test_angles(args, trained_model1, trained_model2, dataloader, attack1):
    # region countersinit
    avg_diff_off_global1 = torch.zeros(args.data_amount)
    avg_diff_on_global1 = torch.zeros(args.data_amount)
    avg_diff_onglobal_offlocal1 = torch.zeros(args.data_amount)
    avg_diff_onglobal_onlocal1 = torch.zeros(args.data_amount)
    avg_diff_total1 = torch.zeros(args.data_amount)

    avg_diff_off_global2 = torch.zeros(args.data_amount)
    avg_diff_on_global2 = torch.zeros(args.data_amount)
    avg_diff_onglobal_offlocal2 = torch.zeros(args.data_amount)
    avg_diff_onglobal_onlocal2 = torch.zeros(args.data_amount)
    avg_diff_total2 = torch.zeros(args.data_amount)

    avg_angle_off_global = torch.zeros(args.data_amount)
    avg_angle_on_global = torch.zeros(args.data_amount)
    avg_angle_onglobal_offlocal = torch.zeros(args.data_amount)
    avg_angle_onglobal_onlocal = torch.zeros(args.data_amount)
    avg_angle_total = torch.zeros(args.data_amount)

    total_as1 = 0
    total_as2 = 0
    avg_data_norm = torch.zeros(args.data_amount)
    # endregion

    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.cuda()
        target = target.cuda()
        model1_classification = trained_model1(data)
        model2_classification = trained_model2(data)
        # print("current classification: {}".format(model1_classification))
        # print("label", target)

        adv_m1 = attack1.attack(trained_model1, data, target)
        adv_m2 = attack1.attack(trained_model2, data, target)
        # print("adv classification ", trained_model(adv_m1).sign())
        if model1_classification.sign() != trained_model1(adv_m1).sign():
            total_as1 += 1

        if model2_classification.sign() != trained_model2(adv_m2).sign():
            # print(model1_classification, target, trained_model1(adv_m1), trained_model2(adv_m2))
            total_as2 += 1

        avg_data_norm[batch_idx] = torch.norm(data, p=2)
        adv1_diff = adv_m1 - data
        adv2_diff = adv_m2 - data

        adv_norm1, adv_norm2, diff_angle = get_projected_norms_and_angles(adv1_diff, adv2_diff)
        avg_diff_total1[batch_idx] = adv_norm1
        avg_diff_total2[batch_idx] = adv_norm2
        avg_angle_total[batch_idx] = diff_angle

        on_global_manifold_diff1norm, on_global_manifold_diff2norm, on_global_manifold_angle = \
            get_projected_norms_and_angles(adv1_diff, adv2_diff, lambda x: project_diff_on_global_basis(x, data))
        avg_diff_on_global1[batch_idx] = on_global_manifold_diff1norm
        avg_diff_on_global2[batch_idx] = on_global_manifold_diff2norm
        avg_angle_on_global[batch_idx] = on_global_manifold_angle

        off_global_manifold_diffnorm1, off_global_manifold_diffnorm2, off_global_manifold_angle = \
            get_projected_norms_and_angles(adv1_diff, adv2_diff, lambda x: project_diff_off_global_basis(x, data))
        avg_diff_off_global1[batch_idx] = off_global_manifold_diffnorm1
        avg_diff_off_global2[batch_idx] = off_global_manifold_diffnorm2
        avg_angle_off_global[batch_idx] = off_global_manifold_angle

        adv_m1_proj_on_globalnlocal_diffnorm, adv_m2_proj_on_globalnlocal_diffnorm, on_globalnlocal_angle = \
            get_projected_norms_and_angles(adv1_diff, adv2_diff, lambda x: project_diff_on_local_on_global_basis(x, data))
        avg_diff_onglobal_onlocal1[batch_idx] = adv_m1_proj_on_globalnlocal_diffnorm
        avg_diff_onglobal_onlocal2[batch_idx] = adv_m2_proj_on_globalnlocal_diffnorm
        avg_angle_onglobal_onlocal[batch_idx] = on_globalnlocal_angle

        adv_m1_on_global_offlocal_diffnorm, adv_m2_on_global_offlocal_diffnorm, on_global_offlocal_angle = \
            get_projected_norms_and_angles(adv1_diff, adv2_diff,
                                           lambda x: project_diff_on_global_off_local_basis(x, data))
        avg_diff_onglobal_offlocal1[batch_idx] = adv_m1_on_global_offlocal_diffnorm
        avg_diff_onglobal_offlocal2[batch_idx] = adv_m2_on_global_offlocal_diffnorm

        avg_angle_onglobal_offlocal[batch_idx] = on_global_offlocal_angle

    #region diffprint
    print("attack success rate1", total_as1, "/10")
    print("attack success rate2", total_as2, "/10")

    print("avg_diff_on_global1 mean", torch.mean(avg_diff_on_global1[avg_diff_on_global1 != 0]))
    print("avg_diff_off_global1 mean", torch.mean(avg_diff_off_global1[avg_diff_off_global1 != 0]))
    print("avg_diff_onglobal_onlocal1 mean",
          torch.mean(avg_diff_onglobal_onlocal1[avg_diff_onglobal_onlocal1 != 0]))
    print("avg_diff_onglobal_offlocal1 mean",
          torch.mean(avg_diff_onglobal_offlocal1[avg_diff_onglobal_offlocal1 != 0]))
    print("avg_diff_total1 mean", torch.mean(avg_diff_total1[avg_diff_total1 != 0]))

    print("avg_diff_on_global2 mean", torch.mean(avg_diff_on_global2[avg_diff_on_global2 != 0]))
    print("avg_diff_off_global2 mean", torch.mean(avg_diff_off_global2[avg_diff_off_global2 != 0]))
    print("avg_diff_onglobal_onlocal2 mean",
          torch.mean(avg_diff_onglobal_onlocal2[avg_diff_onglobal_onlocal2 != 0]))
    print("avg_diff_onglobal_offlocal2 mean",
          torch.mean(avg_diff_onglobal_offlocal2[avg_diff_onglobal_offlocal2 != 0]))
    print("avg_diff_total2 mean", torch.mean(avg_diff_total2[avg_diff_total2 != 0]))
    print("avg_angle_on_global mean", torch.mean(avg_angle_on_global[avg_angle_on_global!=0]))
    print("avg_angle_off_global mean", torch.mean(avg_angle_off_global[avg_angle_off_global!=0]))
    print("avg_angle_onglobal_onlocal mean", torch.mean(avg_angle_onglobal_onlocal[avg_angle_onglobal_onlocal!=0]))
    print("avg_angle_onglobal_offlocal mean", torch.mean(avg_angle_onglobal_offlocal[avg_angle_onglobal_offlocal!=0]))
    print("avg_angle_total mean", torch.mean(avg_angle_total[avg_angle_total!=0]))
    #endregion
