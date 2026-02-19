import __context__

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from algorithms.chebyshev_nystrom import chebyshev_approximation
from algorithms.helpers import gaussian_kernel

plt.style.use("paper/plots/stylesheet.mplstyle")

exponent_formatter = matplotlib.ticker.FuncFormatter(lambda x, _: f"$10^{{{int(np.log10(x)):d}}}$")

m_list = np.logspace(1, 5, 33).astype(int)
sigma_list = np.logspace(-4, 0, 33)

errors = np.ones((len(sigma_list), len(m_list)))

n_t = 1000
t_lin = np.linspace(-1, 1, n_t)
for i, sigma in enumerate(sigma_list):
    kernel = lambda t, x: gaussian_kernel(t, x, sigma=sigma)
    for j, m in enumerate(m_list):
        # Shift bell-curve by a tiny bit to not "coincidentally" get zero error
        mu = chebyshev_approximation(kernel, 1e-1 / 3, m)
        approx = np.polynomial.chebyshev.Chebyshev(mu)(t_lin)
        errors[i, j] = np.max(np.abs(kernel(t_lin, 1e-1 / 3) - approx))

fig, ax = plt.subplots(figsize=(4, 3))

cbar = plt.contourf(range(len(m_list)), range(len(sigma_list)), errors, levels=np.logspace(-18, 4, 23), cmap="plasma", norm=matplotlib.colors.LogNorm(vmin=1e-18, vmax=1e4))

# Add contour for 1e-6
plt.contour(range(len(m_list)), range(len(sigma_list)), errors, levels=[1e-6], colors="black", linestyles="dashed")

# "Dummy"-plot for legend entry
plt.plot([0], [0], color="black", linestyle="dashed", label="$10^{-6}$ contour")

ax.set_xlabel(r"$m$")
ax.set_ylabel(r"$\sigma$")

ax.set_yticks(range(len(sigma_list))[::8])
ax.set_yticklabels([exponent_formatter(val, None) for val in sigma_list[::8]])
ax.set_xticks(range(len(m_list))[::8])
ax.set_xticklabels([exponent_formatter(val, None) for val in m_list[::8]])

bound = lambda sigma, error: (np.log(2 * np.sqrt(np.e * 2 / np.pi)) - np.log(sigma ** 2 * error)) / np.log(1 + sigma)
sigma_bound = np.log10(sigma_list) / np.log10(sigma_list[0]) * (33 - 1)
m_bound = np.log10(bound(sigma_list[::-1], 1e-6)) / np.log10(m_list[-1]) * (33 - 1)

#plt.plot(m_bound, sigma_bound, color="black")
plt.fill_betweenx(sigma_bound, m_bound, 32, color="black", hatch="//", alpha=0.3, label=r"$E_{\sigma, m} \leq 10^{-6}$")
#plt.text(15.5, 15.5, r"$E_{\sigma, m} \leq 10^{-6}$", color="black", rotation=-48)
plt.xlim(0, 32)

bar = plt.colorbar(cbar, aspect=8)
bar.ax.hlines(1e-6, xmin=0, xmax=2, colors="black", linestyles="dashed")

plt.grid(alpha=1)
plt.legend(loc="lower left")
plt.savefig("paper/plots/chebyshev_heatmap.pgf", bbox_inches="tight")
