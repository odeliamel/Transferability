import torch
from synthetic.utils import get_batch_size
from torch.utils.data import DataLoader


def create_adv_pert(args, batch, output):
    margin_indices = find_margin_indices(args, batch, output)
    data_on_margin = torch.index_select(batch[0], 0, torch.tensor(margin_indices))
    labels_on_margin = torch.index_select(batch[1], 0, torch.tensor(margin_indices))
    z = args.pert_size * sum((data_on_margin.T * labels_on_margin).T)
    z_mat = torch.outer(z, -batch[1]).T
    return z_mat


def find_margin_indices(args, batch, output):
    ordered_points = (output.T * batch[1])[0].tolist()
    ordered_points.sort()
    margin = args.pert_margin_tolerance * 2
    best_low_idx = 0
    best_high_idx = 0
    best_num_of_points = 0
    cur_low_idx = 0
    for i in range(len(ordered_points)):
        if i - best_low_idx + 1 > best_num_of_points and ordered_points[i] - ordered_points[cur_low_idx] < margin:
            best_low_idx = cur_low_idx
            best_high_idx = i
            best_num_of_points = i - cur_low_idx + 1
        else:
            cur_low_idx += 1
    largest_num = ordered_points[best_high_idx]
    smallest_num = ordered_points[best_low_idx]
    indices_list = []

    points = (output.T * batch[1])[0]
    for i in range(len(points)):
        if (points[i] <= largest_num) and (points[i] >= smallest_num):
            indices_list.append(i)

    return indices_list


def evaluate_adv_pert(args, batch, adv_output, output):

    # test the adv pert on the margin samples
    margin_indices = find_margin_indices(args, batch, output)
    data_on_margin = torch.index_select(batch[0], 0, torch.tensor(margin_indices))
    labels_on_margin = torch.index_select(batch[1], 0, torch.tensor(margin_indices))
    adv_output_on_margin = torch.index_select(adv_output, 0, torch.tensor(margin_indices))
    adv_labels = (adv_output_on_margin.T * labels_on_margin).T

def try_adv_pert(args, model, dataset):
    batch_size = get_batch_size(args)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # should be full batch to work correctly
    for batch_idx, batch in enumerate(dataloader):
        output = model(batch[0].float())
        adv_mat = create_adv_pert(args, batch, output)
        adv_output = model((batch[0] + adv_mat).float())
        evaluate_adv_pert(args, batch, adv_output, output)
    x=0

