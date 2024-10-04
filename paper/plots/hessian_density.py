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
    torch.nn.Sigmoid(),
)

np.random.seed(0)

train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST("matrices/mnist_data", 
                                           download=True,
                                           train=True,
                                           transform=torchvision.transforms.ToTensor()),
                                           batch_size=64,
                                           shuffle=False
)

test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST("matrices/mnist_data", 
                                          download=True,
                                          train=False,
                                          transform=torchvision.transforms.ToTensor()),
                                          batch_size=64,
                                          shuffle=False
)


# Set parameter
t = np.linspace(0.05, 1.0, 150)
sigma = 0.005
m = 1000
n_Omega = 10
n_Psi = 10
data = next(iter(train_loader))

plt.style.use("paper/plots/stylesheet.mplstyle")
plt.figure(figsize=(3, 3))

# Train the model on the MNIST data set
loss_function = lambda x, y: torch.nn.MSELoss()(x, torch.nn.functional.one_hot(y, num_classes=10).to(dtype=torch.float))
validator = lambda model: accuracy_validator(model, test_loader)

# Approximate the Hessian's spectral density
kernel = lambda t, x: gaussian_kernel(t, x, sigma=sigma, n=H.shape[0])
H = hessian(model, loss_function, data, spectral_transform=True)
phi = chebyshev_nystrom(H, t, m, n_Psi, n_Omega, kernel)

plt.plot(t * H.scaling_parameter, phi, color="#648FFF", label=r"untrained")

epoch_loss = 0
for batch in train_loader:
    epoch_loss += loss_function(model(*batch[:-1]), batch[-1]).item()

loss_list = [epoch_loss / len(train_loader)]

loss_list.extend(train(model, train_loader, loss_function, validator=validator, n_epochs=2))

H.update(model, loss_function, data)
phi = chebyshev_nystrom(H, t, m, n_Psi, n_Omega, kernel)

plt.plot(t * H.scaling_parameter, phi, color="#785EF0", label=r"epoch $2$")

loss_list.extend(train(model, train_loader, loss_function, validator=validator, n_epochs=2))

H.update(model, loss_function, data)
phi = chebyshev_nystrom(H, t, m, n_Psi, n_Omega, kernel)

plt.plot(t * H.scaling_parameter, phi, color="#DC267F", label=r"epoch $4$")

loss_list.extend(train(model, train_loader, loss_function, validator=validator, n_epochs=2))

H.update(model, loss_function, data)
phi = chebyshev_nystrom(H, t, m, n_Psi, n_Omega, kernel)

plt.plot(t * H.scaling_parameter, phi, color="#FE6100", label=r"epoch $6$")

loss_list.extend(train(model, train_loader, loss_function, validator=validator, n_epochs=2))

H.update(model, loss_function, data)
phi = chebyshev_nystrom(H, t, m, n_Psi, n_Omega, kernel)

plt.plot(t * H.scaling_parameter, phi, color="#FFB000", label=r"epoch $8$")
plt.grid(True, which="both")
plt.ylabel(r"smoothed spectral density $\phi_{\sigma}(t)$")
plt.xlabel(r"spectral parameter $t$")
plt.legend()
plt.savefig("paper/plots/hessian_density__.pgf", bbox_inches="tight")

plt.figure(figsize=(3, 3))
plt.plot(range(len(loss_list)), loss_list, marker="d", color="k")
plt.xticks([0, 2, 4, 6, 8], [0, 2, 4, 6, 8])
plt.grid(True, which="both")
plt.ylabel(r"training loss")
plt.xlabel(r"epoch")
plt.savefig("paper/plots/hessian_density_loss__.pgf", bbox_inches="tight")
