#! /usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

path = './pics_projection_approach'

np.random.seed(0)

n_points = 40
mm = np.array([1, 0, -1, 2])
x = np.linspace(0, 1, n_points)
features = [['a', 'b'], ['i', 'j'], ['m', 'n'], ['x', 'y']]
for idx, m in enumerate(mm):

    if m == 2:
        x = np.random.random(size=n_points)
        y = np.random.random(size=n_points)
        A = [x[n_points//3], y[n_points//3]]
        B = [2*x[n_points//3], 2*y[n_points//3]]
    else:
        y = m*x + np.random.normal(loc=0, scale=0.2, size=n_points)
        if m == 1:
            A = [x[-3], y[-3] - 1.5]
            B = [x[10], y[10]]
        elif m == -1:
            A = [x[10], y[10]]
            B = [x[-3], y[-3] + 1.5]
        else:
            A = [x[10], y[10]]
            B = [x[-3], y[-3]]
    fig, ax = plt.subplots(figsize=(10, 5))
    # ax.axis('equal')
    # ax.set_xlim((-0.3, 1.3))
    # ax.set_xlim((-0.3, 1.3))

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel(f'Feature {features[idx][0]}', fontsize='xx-large')
    ax.set_ylabel(f'Feature {features[idx][1]}', fontsize='xx-large')
    ax.scatter(x, y)
    ax.scatter(A[0], A[1], s=150, label='A')
    ax.scatter(B[0], B[1], s=150, label='B')
    ax.legend(loc='upper left')

    plt.tight_layout()
    fig.savefig(f'{path}/{idx}_projection_approach.png')
    fig.savefig(f'{path}/{idx}_projection_approach.pdf')
