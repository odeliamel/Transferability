import torch
from datasets import dataset
from views import basicview
import os
import matplotlib.pyplot as plt
import pickle

from views.basicview import ax_color_image_view, ax_bw_image_view

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# from sklearn.decomposition import PCA


def preproccess():
    train_set = dataset.MNIST.getTrainSetIterator(batch_size=10000)

    data_as_matrix = torch.tensor([])
    for data, target in train_set:
        data = (data - torch.tensor((0.1307,))) / torch.tensor((0.3081,))
        data_as_matrix = torch.cat([data_as_matrix, data.reshape(data.shape[0], 784)], dim=0)
        print(data_as_matrix.shape)

    U ,S ,V = torch.pca_lowrank(data_as_matrix, q=784, niter=10)
    k = 50
    print(U.shape, S.shape, V.shape)
    print(S)
    plt.plot(S)
    plt.show()
    pca_dict = {"U": U, "S": S, "V": V}
    new_file = os.path.join('PCAs', 'pca784q10iters.pickle')
    with open(new_file, 'wb') as handle:
        pickle.dump(pca_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    zresult = torch.matmul(data_as_matrix, V[:, :k])
    print(zresult.shape, V[:, :k].shape)
    xapprox = torch.matmul(V[:, :k], zresult.T)
    print(xapprox.shape)
    xapprox = xapprox.T
    img = torch.reshape(xapprox[0], (1, 28, 28))
    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 1)
    ax_bw_image_view(ax, torch.reshape(data_as_matrix[0], (1, 28, 28)))

    ax = fig.add_subplot(1, 2, 2)
    ax_bw_image_view(ax, img)


    plt.show()


if __name__ == '__main__':
    preproccess()
