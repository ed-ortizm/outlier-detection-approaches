import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

def lp(x, y, p=2):
# https://en.wikipedia.org/wiki/Lp_space#The_p-norm_in_finite_dimensions

    d = np.sum((np.abs(x-y))**p)**(1/p)
    return d

def nns(data, n_neighbors=2, p=2, n_jobs=6):

    NNs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric=lp, metric_params={'p': p}, n_jobs=n_jobs)
    NNs_data = NNs.fit(data)
    dd, idxx = NNs_data.kneighbors(data)

    return dd[:, 1:], idxx[:, 1:]


def plt_data(data, fname, path='./', alpha=0.2, face_color = True):

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_ylabel('$y$')
    ax.set_xlabel('$x$')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.axis('off')

    ax.scatter(data[:, 0], data[:, 1], alpha=alpha, color='blue')
    plt.tight_layout()

    fig.savefig(f'{path}/{fname}.pdf')
    if face_color:
        fig.savefig(f'{path}/{fname}.png')
    else:
        fig.savefig(f'{path}/{fname}.png', facecolor='none')
