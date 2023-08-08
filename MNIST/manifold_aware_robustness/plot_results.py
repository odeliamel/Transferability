from turtledemo.__main__ import font_sizes

# final sphere dim 32
s = """{'identity': [[1.6269, 1.6081, 1.6986, 1.6556, 1.6633],
        [2.3574, 2.2848, 2.3262, 2.3206, 2.3115],
        [3.8475, 3.8487, 3.9676, 3.7952, 3.7875],
        [4.2031, 4.2958, 4.2931, 3.9997, 4.1507],
        [4.2564, 4.3636, 4.3621, 4.1006, 4.2449]], 'project_diff_on_global_basis': [[5.0155, 5.2940, 5.2914, 4.9875, 5.2425],
        [4.7097, 4.8255, 4.9183, 4.5741, 4.7224],
        [4.4373, 4.5615, 4.6745, 4.3498, 4.4216],
        [4.4397, 4.5694, 4.5450, 4.2268, 4.3901],
        [4.3744, 4.5018, 4.4936, 4.2065, 4.3680]], 'project_diff_off_global_basis': [[ 1.7011,  1.6817,  1.7827,  1.7451,  1.7415],
        [ 2.6848,  2.5586,  2.6063,  2.6594,  2.6033],
        [ 7.4764,  6.9873,  7.3177,  7.5323,  7.1308],
        [12.4824, 12.1588, 12.7243, 12.0679, 12.2346],
        [17.6323, 17.0560, 17.7615, 17.6184, 17.1913]]}"""


#final MNIST
# s = """{'identity': [[0.7265, 0.7238, 0.7503, 0.7406, 0.7578],
#         [0.9720, 0.9682, 1.0095, 0.9597, 0.9266],
#         [1.0965, 1.0522, 1.1029, 1.0830, 1.0657],
#         [1.1719, 1.0681, 1.0964, 1.0882, 1.1323],
#         [1.0860, 1.2665, 1.0923, 1.1108, 1.2283],
#         [1.1319, 1.1410, 1.2164, 1.2649, 1.2167]], 'project_diff_on_global_basis': [[1.3961, 1.3655, 1.4267, 1.4051, 1.4358],
#         [1.3681, 1.3663, 1.4106, 1.3490, 1.3004],
#         [1.2736, 1.2250, 1.2813, 1.2565, 1.2359],
#         [1.2378, 1.1317, 1.1684, 1.1521, 1.1938],
#         [1.1411, 1.3303, 1.1487, 1.1694, 1.2910],
#         [1.1915, 1.1910, 1.2753, 1.3266, 1.2756]], 'project_diff_off_global_basis': [[0.8596, 0.8674, 0.8939, 0.8790, 0.9031],
#         [1.4271, 1.4164, 1.4828, 1.4023, 1.3550],
#         [2.3945, 2.2546, 2.3698, 2.3271, 2.2935],
#         [5.4057, 4.8702, 5.0617, 5.0246, 5.2711],
#         [6.1746, 7.4635, 6.3674, 6.5076, 7.3008],
#         [7.1932, 7.4132, 7.9624, 8.2219, 7.8767]]}"""


distances = eval(s)
factors = [0.5] + list(range(1, 8, 2))
# factors = [0.2, 0.5] + list(range(1, 8, 2))
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
max_distances = {"identity": [], "project_diff_on_global_basis": [], "project_diff_off_global_basis": []}
min_distances = {"identity": [], "project_diff_on_global_basis": [], "project_diff_off_global_basis": []}
mean_distances = {"identity": [], "project_diff_on_global_basis": [], "project_diff_off_global_basis": []}
for key in distances.keys():
    print("1", distances[key])
    for i in range(len(factors)):
        print("2", distances[key][i])
        tmax = max(distances[key][i])
        tmin = min(distances[key][i])
        tmean = mean(distances[key][i])
        mean_distances[key].append(tmean)
        max_distances[key].append(tmax - tmean)
        min_distances[key].append(tmean - tmin)
print(max_distances)
lbl = ["In all input space", "On data subspace", "Off data subspace"]
i = 0
fig = plt.figure()
fig.set_size_inches(12, 8)
for key in distances.keys():
    # plt.plot(factors, distances[key], label=key)
    # plt.plot(factors, distances[key], label=key)
    print(mean_distances[key])
    plt.errorbar(factors, mean_distances[key], yerr=[min_distances[key], max_distances[key]], label=lbl[i], ecolor='black', capsize=2)
    i += 1
plt.legend(fontsize=14)
# plt.xticks([0.2, 0.5] + list(np.arange(1, 8, step=1)))
plt.xticks([0.5] + list(np.arange(1, 8, step=1)))
# plt.xlim((0.1, 7.1))
plt.xlim((0.4, 7.1))
# plt.yticks(np.arange(0, 9, step=1))
plt.yticks(np.arange(0, 19, step=1))
# plt.title("The average perturbation size in different subspaces")
plt.show()
