import __context__

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from algorithms.chebyshev_nystrom import chebyshev_nystrom
from algorithms.helpers import gaussian_kernel
from matrices.neural_network import model, train, accuracy_validator, hessian

# For now, don't execute this
if __name__ == "__main__":
    exit()

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

# Train the model on the MNIST data set
loss_function = lambda x, y: torch.nn.MSELoss()(x, torch.nn.functional.one_hot(y, num_classes=10).to(dtype=torch.float))
validator = lambda model: accuracy_validator(model, test_loader)
train(model, train_loader, loss_function, validator=validator, n_epochs=30)

data = next(iter(train_loader))
H = hessian(model, loss_function, data, spectral_transform=True)

# Set parameter
t = np.linspace(-0.1, 0.1, 100)
sigma = 0.02
m = 700
n_Omega = 80
n_Psi = 20

plt.style.use("paper/plots/stylesheet.mplstyle")

# Approximate the Hessian's spectral density
kernel = lambda x: gaussian_kernel(x, sigma=sigma)
phi = chebyshev_nystrom(H, t, m, n_Psi, n_Omega, kernel)

plt.plot(t, phi, color="#648FFF")

plt.grid(True, which="both")
plt.ylabel(r"smoothed spectral density $\phi_{\sigma}(t)$")
plt.xlabel(r"spectral parameter $t$")
plt.savefig("paper/plots/hessian_density.pgf", bbox_inches="tight")
