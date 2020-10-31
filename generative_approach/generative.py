#! /usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

path = './pics_generative_approach'
np.random.seed(0)

n_points = 10_000

mus = np.array([2, 3, 5, 7])
sigmas = np.random.random(size=mus.size)
x = np.linspace(0, 9, n_points)
models = np.empty((sigmas.size, n_points))

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlabel('Observables', fontsize='xx-large')
ax.set_ylabel('Measurements', fontsize='xx-large')
ax.set_yticklabels([])
ax.set_xticklabels([])

lines = []
for idx, mu in enumerate(mus):
    models[idx, :] = norm.pdf(x, loc=mu, scale=sigmas[idx])
    model_line, = ax.plot(x, models[idx, :], alpha=0.12)
    # lines.append(model_line)

legends = [f'Model {idx+1}' for idx in range(len(lines))]

# Final model
model = np.sum(models, axis=0)
line, = ax.plot(x, model, color='black', alpha=5)
lines.append(line)
legends.append('Generative process')

ax.legend(lines, legends, loc='upper right', shadow=True)

# Deviations

noise = 0.5 - np.random.random(size=n_points)
data = model + noise*0.7
idx = np.random.randint(0, n_points, int(n_points*0.015))

ax.scatter(x[idx], data[idx], s=2.)
plt.tight_layout()
fig.savefig(f'{path}/generative_models.png')
fig.savefig(f'{path}/generative_models.pdf')
