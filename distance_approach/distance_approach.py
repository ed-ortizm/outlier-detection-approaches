#! /usr/bin/env python3
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors

from lib_mg2020 import plt_data

np.random.seed(0)

# https://en.wikipedia.org/wiki/Lp_space#The_p-norm_in_finite_dimensions

def lp(x, y, p=2):

    d = np.sum((np.abs(x-y))**p)**(1/p)
    return d

def nns(data, nns=2, p=2):

    NNs = NearestNeighbors(n_neighbors=nns, algorithm='auto', metric=lp, metric_params={'p': p})
    NNs_data = NNs.fit(data)
    dd, idxx = NNs_data.kneighbors(X)

    return dd[:, 1:], idxx[:, 1:]

path = './pics_test_mg2020'

ti = time.time()

## intuition data
# inliers

n_samples = np.array([100, 100, 100])
# n_samples = np.array([1_000, 2_000, 600])
centers = np.array([[2, 2], [3, 3], [2.5, 1]])
cluster_std = [1, 0.1, 1]
# cluster_std = [0.1, 0.1, 0.1]
X, y= make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=1)

data = X
print(f'data shape: {data.shape}')

# outliers

n_outliers = 2
ox1 = 1.3 + np.random.random(size= n_outliers)*0.3
oy1 = 0.5 + np.random.random(size= n_outliers)*3
data = np.vstack((data, np.vstack((ox1, oy1)).T))
print(f'data shape: {data.shape}')

n_outliers = 6
ox2 = 1.6 + np.random.random(size= n_outliers)*0.55
oy2 = 0.5 + np.random.random(size= n_outliers)
data = np.vstack((data, np.vstack((ox2, oy2)).T))
print(f'data shape: {data.shape}')

n_outliers = 4
ox3 = 1.6 + np.random.random(size= n_outliers)*0.8
oy3 = 2.6 + np.random.random(size= n_outliers)*0.9
data = np.vstack((data, np.vstack((ox3, oy3)).T))
print(f'data shape: {data.shape}')

n_outliers = 2
ox4 = 2.2 + np.random.random(size= n_outliers)*1.2
oy4 = 0.5 + np.random.random(size= n_outliers)*0.05
data = np.vstack((data, np.vstack((ox4, oy4)).T))
print(f'data shape: {data.shape}')


n_outliers = 5
ox5 = 2.4 + np.random.random(size= n_outliers)*0.2
oy5 = 1.5 + np.random.random(size= n_outliers)*2
data = np.vstack((data, np.vstack((ox5, oy5)).T))
print(f'data shape: {data.shape}')

n_outliers = 5
ox6 = 2.6 + np.random.random(size= n_outliers)*0.8
oy6 = 1.5 + np.random.random(size= n_outliers)
data = np.vstack((data, np.vstack((ox6, oy6)).T))
print(f'data shape: {data.shape}')

n_outliers = 2
ox7 = 3 + np.random.random(size= n_outliers)*0.4
oy7 = 0.5 + np.random.random(size= n_outliers)
data = np.vstack((data, np.vstack((ox7, oy7)).T))
print(f'data shape: {data.shape}')

# cluster outlier
sigma = 0.05
size = 12
ox0 = np.random.normal(loc=1.6, scale=sigma, size=size)
oy0 = np.random.normal(loc=3.0, scale=sigma, size=size)
data = np.vstack((data, np.vstack((ox0, oy0)).T))
print(f'data shape: {data.shape}')



##NNs
dd, idxx = nns(data=data, nns=data.shape[0], p=2)

np.save('distances_p_2.npy', dd)

plt_data(data)
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
