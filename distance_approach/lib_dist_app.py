import matplotlib.pyplot as plt
import numpy as np

def plt_data(data, face_color = True):

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_ylabel('$y$')
    ax.set_xlabel('$x$')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.axis('off')

    ax.scatter(data[:, 0], data[:, 1], alpha=0.07, color='blue')
    plt.tight_layout()

    fig.savefig('outlier_intuitive.pdf')
    if face_color:
        fig.savefig('outlier_intuitive.png')
    else:
        fig.savefig('outlier_intuitive.png', facecolor='none')
