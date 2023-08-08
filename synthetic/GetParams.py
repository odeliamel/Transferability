import argparse


def get_args(*args):
    parser = argparse.ArgumentParser(description='Adversarial Perturbations')

    # general parameters
    parser.add_argument('--cuda', default='False', type=str2bool, help='')
    parser.add_argument('--seed', default=1, type=int, help='')

    # data creation
    parser.add_argument('--data_type', default='uniform', help='options: gaussian, uniform')
    parser.add_argument('--input_dim', default=28*28, type=int, help='')
    parser.add_argument('--data_dim', default=50, type=int, help='')
    parser.add_argument('--data_amount', default=100000, type=int, help='')
    parser.add_argument('--data_labels', default='random', help='options: random, ')
    parser.add_argument('--data_uniform_norm', default=(28*28)**0.5, help='0 is sqrt(data_dim)')
    parser.add_argument('--data_cluster_size', default=1, help='how many data points are in this cluster')
    parser.add_argument('--data_cluster_radius', default=0, help='the clusters radius')



    # model
    parser.add_argument('--model_hidden_dim', default=28*28, type=int, help='')
    parser.add_argument('--model_activation', default='relu', help='options: relu')

    # train
    parser.add_argument('--train_iterations', default=6000, type=int, help='')
    parser.add_argument('--train_lr', default=0.0001, type=float, help='')
    parser.add_argument('--train_batch_size', default=2000, type=int, help='0 is batch size = data amount')
    parser.add_argument('--train_loss', default='log', help='options: exp, log, square')
    parser.add_argument('--train_evaluate_rate', default=10, type=int, help='')

    # adv perturbation
    parser.add_argument('--pert_size', default=1., type=float, help='')
    parser.add_argument('--pert_margin_tolerance', default=1, type=float, help='')
    parser.add_argument('--adv_train_projection', default=None, type=float, help='')
    parser.add_argument('--adv_loss', default='margin', help='')



    if not isinstance(args, list):
        args = args[0]
    args = parser.parse_args(args)
    return args


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')