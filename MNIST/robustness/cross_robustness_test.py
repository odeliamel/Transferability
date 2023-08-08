import time
import torch


def cross_test_w_dist(model, attack, sur_model, condition=None, num_restarts=1):
    if condition is None:
        if attack.targeted:
            condition = lambda classification, t: classification == t
        else:
            condition = lambda classification, t: classification != t

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 100
    total_failed_attacks = torch.ones(10000, device=device, dtype=torch.bool)
    if model.dataset.__name__ == "IMAGENET":
        total_failed_attacks = torch.ones(5000, device=device, dtype=torch.bool)
    distances = torch.zeros_like(total_failed_attacks, dtype=torch.float)
    # handling restarts efficiently
    startt = time.time()
    for i, [data, target] in enumerate(model.dataset.getTestSetIterator(batch_size=batch_size)):
        bstart = i * batch_size
        bend = (i + 1) * batch_size
        if model.dataset.__name__ == "IMAGENET" and bstart >= 5000:
            break
        # print(bstart, bend)
        data, target = data.to(device), target.to(device)
        for c in range(num_restarts):
            current_to_improve = total_failed_attacks[bstart:bend].clone().type(torch.bool)

            X_adv, batch_targets = \
                attack.attack_batch_w_targets(model, data[current_to_improve], target[current_to_improve])

            acc_each = condition(sur_model(X_adv), batch_targets).cuda()# & condition(model(X_adv).data.max(1)[1], batch_targets).cuda()

            distances[bstart:bend][current_to_improve] = total_failed_attacks[bstart:bend][current_to_improve].float() * acc_each.float() * \
                                                         torch.norm((X_adv - data[current_to_improve]), p=2,
                                                                    dim=[1, 2, 3])
            total_failed_attacks[bstart:bend][current_to_improve] = (total_failed_attacks[bstart:bend][current_to_improve] * ~acc_each)

            # print(total_failed_attacks[bstart:bend])
            # print(distances[~total_failed_attacks].max())
        #     break
        # break
    asr = (~total_failed_attacks).sum().float() / total_failed_attacks.shape[0]
    print('asr_total: {:.2%}'.format(asr))
    success_dist = distances[~total_failed_attacks]
    # print("success dist: ", success_dist, success_dist.shape)
    mean_eps = success_dist.mean()
    print("max eps: {:.4f}".format(success_dist.max()))
    print("mean eps: {:.4f}".format(mean_eps))
    total_time = time.time() - startt
    print("attack time {:.1f} s".format(total_time))
    model.zero_grad()
    # return asr, mean_eps, total_time
    return asr

