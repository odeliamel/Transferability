import numpy as np
from torch.utils.data import Dataset



class CustomDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.dataset = x
        self.labels = y


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]


def create_data(args):
    # construct data
    if args.data_type == 'gaussian':
        data = create_gaussian_data(args)
    elif args.data_type == 'uniform':
        data = create_uniform_data(args)

    # construct lables
    if args.data_labels == 'random':
        labels = (np.random.randint(2, size=len(data)) - 0.5) * 2
        labels = labels.astype(np.int64)

    dataset = CustomDataset(data, labels)

    return dataset


def create_gaussian_data(args):
    data = []
    input_dim = args.data_dim
    for i in range(args.data_amount):
        new_data_point = np.random.normal(0, input_dim ** 0.5, size=input_dim)
        data.append(new_data_point)
    return data


def create_uniform_data(args):
    data = []
    data_dim = args.data_dim
    if args.data_uniform_norm == 0:
        data_norm = args.data_dim ** 0.5
    else:
        data_norm = args.data_uniform_norm
    for i in range(args.data_amount):
        new_data_point = np.zeros(args.input_dim, dtype=np.float32)
        new_data_point[0:data_dim] = np.random.normal(0, 1, size=data_dim)
        new_data_point = data_norm * (new_data_point / np.linalg.norm(new_data_point, ord=2))
        for i in range(args.data_cluster_size):
            point = new_data_point.copy()
            point[0:data_dim] += np.random.normal(0, args.data_cluster_radius, size=data_dim)
            data.append(point)
    print(len(data))
    return data