"""
hessian_density.py

TODO: Only look at spectrum at range far in the negative or positive
"""

import __context__

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from algorithms.chebyshev_nystrom import chebyshev_nystrom
from algorithms.helpers import gaussian_kernel
from matrices.neural_network import model, train, accuracy_validator, hessian

model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 1, kernel_size=(3, 3)),
    torch.nn.Softplus(),
    torch.nn.BatchNorm2d(1),
    torch.nn.Flatten(),
    torch.nn.Linear(676, 10),
    #torch.nn.Softplus(),
    #torch.nn.Linear(256, 10),
    torch.nn.Sigmoid(),
)

# For now, don't execute this
#if __name__ == "__main__":
#    exit()

np.random.seed(0)

train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST("matrices/mnist_data", 
                                           download=True,
                                           train=True,
                                           transform=torchvision.transforms.ToTensor()),
                                           batch_size=64,
                                           shuffle=True
)

test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST("matrices/mnist_data", 
                                          download=True,
                                          train=False,
                                          transform=torchvision.transforms.ToTensor()),
                                          batch_size=64,
                                          shuffle=False
)


# Set parameter
t = np.linspace(0.2, 1.0, 100)
sigma = 0.005
m = 500
n_Omega = 30
n_Psi = 10
data = next(iter(train_loader))

plt.style.use("paper/plots/stylesheet.mplstyle")

# Train the model on the MNIST data set
loss_function = lambda x, y: torch.nn.MSELoss()(x, torch.nn.functional.one_hot(y, num_classes=10).to(dtype=torch.float))
validator = lambda model: accuracy_validator(model, test_loader)

# Approximate the Hessian's spectral density
kernel = lambda t, x: gaussian_kernel(t, x, sigma=sigma, n=H.shape[0])
H = hessian(model, loss_function, data, spectral_transform=True)
phi = chebyshev_nystrom(H, t, m, n_Psi, n_Omega, kernel)

plt.plot(t, phi, color="#648FFF", label=r"untrained")

train(model, train_loader, loss_function, validator=validator, n_epochs=5)

H.update(model, loss_function, data)
phi = chebyshev_nystrom(H, t, m, n_Psi, n_Omega, kernel)

plt.plot(t, phi, color="#DC267F", label=r"epoch $5$")

train(model, train_loader, loss_function, validator=validator, n_epochs=5)

H.update(model, loss_function, data)
phi = chebyshev_nystrom(H, t, m, n_Psi, n_Omega, kernel)

plt.plot(t, phi, color="#FFB000", label=r"epoch $10$")
plt.grid(True, which="both")
plt.ylabel(r"smoothed spectral density $\phi_{\sigma}(t)$")
plt.xlabel(r"spectral parameter $t$")
plt.legend()
plt.savefig("paper/plots/hessian_density.pgf", bbox_inches="tight")
