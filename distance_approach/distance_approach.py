#! /usr/bin/env python3
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.datasets import make_blobs


from lib_dist_app import plt_data, nns

ti = time.time()

np.random.seed(0)

## intuition data
# inliers

n_samples = [100, 100, 100, 100, 150, 51]
centers = [ [5, 5], [5, 9], [9, 5], [5, 1], [2, 5], [-5, -2] ]
cluster_std = [0.25, 0.5, 1, 1.5, 3, 0.25]
data, y= make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=1)

##NNs
p = 2
n_neighbors = data.shape[0]

dd, idxx = nns(data=data, n_neighbors=n_neighbors, p=p)

np.save(f'distances_p_{p}.npy', dd)

rr = [1, 50, 100, 200, 400, 600]

for r in rr:

    fig, ax = plt.subplots(figsize=(10,5))

    d_r = np.mean(dd[:, :r], axis=1)
    d_r /= np.max(d_r)

    ax.hist(d_r, bins=100)
    plt.tight_layout()

    fig.savefig(f'nns_{r}_outlier_score.png')

    plt.close()


# plt_data(data=data, fname='data_distribution', face_color = True)
#
# fig, ax = plt.subplots(figsize=(12, 8))
# dd_sorted = np.sort(dd)
# ax.hist(dd_sorted[:1_000], bins=100)
# plt.tight_layout()
#
# fig.savefig('hist_outlier_intuitive.pdf')
# fig.savefig('hist_outlier_intuitive.png', facecolor='none')



# # silly example
# X = np.array([[1, 1], [2, 1], [3, 2]])
#
# dd, idxx = nns(data=X, nns=3, p=2)
#
# print(X, '\n', dd, '\n', idxx, '\n')



#############################################################################
# n_points = 40
# mm = np.array([1, 0, -1, 2])
# x = np.linspace(0, 1, n_points)
# features = [['a', 'b'], ['i', 'j'], ['m', 'n'], ['x', 'y']]
# for idx, m in enumerate(mm):
#
#     if m == 2:
#         x = np.random.random(size=n_points)
#         y = np.random.random(size=n_points)
#         A = [x[n_points//3], y[n_points//3]]
#         B = [2*x[n_points//3], 2*y[n_points//3]]
#     else:
#         y = m*x + np.random.normal(loc=0, scale=0.2, size=n_points)
#         if m == 1:
#             A = [x[-3], y[-3] - 1.5]
#             B = [x[10], y[10]]
#         elif m == -1:
#             A = [x[10], y[10]]
#             B = [x[-3], y[-3] + 1.5]
#         else:
#             A = [x[10], y[10]]
#             B = [x[-3], y[-3]]
#     fig, ax = plt.subplots(figsize=(10, 5))
#     # ax.axis('equal')
#     # ax.set_xlim((-0.3, 1.3))
#     # ax.set_xlim((-0.3, 1.3))
#
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.set_xlabel(f'Feature {features[idx][0]}', fontsize='xx-large')
#     ax.set_ylabel(f'Feature {features[idx][1]}', fontsize='xx-large')
#     ax.scatter(x, y)
#     ax.scatter(A[0], A[1], s=150, label='A')
#     ax.scatter(B[0], B[1], s=150, label='B')
#     ax.legend(loc='upper left')
#
#     plt.tight_layout()
#     fig.savefig(f'{path}/{idx}_projection_approach.png')
#     fig.savefig(f'{path}/{idx}_projection_approach.pdf')
# ####################################################
# path = './pics_generative_approach'
# np.random.seed(0)
#
# n_points = 10_000
#
# mus = np.array([2, 3, 5, 7])
# sigmas = np.random.random(size=mus.size)
# x = np.linspace(0, 9, n_points)
# models = np.empty((sigmas.size, n_points))
#
# fig, ax = plt.subplots(figsize=(10, 5))
# ax.set_xlabel('Observables', fontsize='xx-large')
# ax.set_ylabel('Measurements', fontsize='xx-large')
# ax.set_yticklabels([])
# ax.set_xticklabels([])
#
# lines = []
# for idx, mu in enumerate(mus):
#     models[idx, :] = norm.pdf(x, loc=mu, scale=sigmas[idx])
#     model_line, = ax.plot(x, models[idx, :], alpha=0.12)
#     # lines.append(model_line)
#
# legends = [f'Model {idx+1}' for idx in range(len(lines))]
#
# # Final model
# model = np.sum(models, axis=0)
# line, = ax.plot(x, model, color='black', alpha=5)
# lines.append(line)
# legends.append('Generative process')
#
# ax.legend(lines, legends, loc='upper right', shadow=True)
#
# # Deviations
#
# noise = 0.5 - np.random.random(size=n_points)
# data = model + noise*0.7
# idx = np.random.randint(0, n_points, int(n_points*0.015))
#
# ax.scatter(x[idx], data[idx], s=2.)
# plt.tight_layout()
# fig.savefig(f'{path}/generative_models.png')
# fig.savefig(f'{path}/generative_models.pdf')
tf = time.time()

print(f'Running time: {tf-ti:.2f}')
