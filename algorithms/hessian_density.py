"""
hessian_density.py
------------------

Spectral density of Hessian matrix.
"""

import torch
import time
import numpy as np
from .krylov_aware import lanczos

def compute_jacobian(model, loss_function, data):
    input, target = data
    output = model(input)
    loss = loss_function(output, target)

    jac = torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
    jac = torch.cat([e.ravel() for e in jac])
    return jac

def hessian_vector_product(jacobian, params, X):
    return torch.autograd.grad(jacobian, params, X, retain_graph=True, is_grads_batched=True if X.ndim > 1 else False)

class hessian():
    def __init__(self, model, loss_function, data, spectral_transform=True):
        self.parameters = list(model.parameters())
        self.num_parameters = sum(p.numel() for p in model.parameters())
        self.shape = (self.num_parameters, self.num_parameters)
        self.jacobian = compute_jacobian(model, loss_function, data)
        self.scaling_parameter = 1.0
        if spectral_transform:
            k = 20
            x = torch.randn(self.num_parameters)
            Q, T = lanczos(self, x, k, dtype=np.float64)
            self.scaling_parameter = np.max(np.linalg.eigvalsh(T[0, :-1, :])) + np.linalg.norm(T[0, -1, -1] * Q[:, :, -1])

    def __matmul__(self, X):
        is_numpy = isinstance(X, np.ndarray)
        if not torch.is_tensor(X):
            X = torch.from_numpy(X)
        hvp = hessian_vector_product(self.jacobian, self.parameters, X.T)
        hvp = torch.cat([e.reshape(X.shape[-1], -1) for e in hvp], dim=1).T
        if is_numpy:
            hvp = hvp.numpy()
        return hvp / self.scaling_parameter


def train(model, data_loader, loss_function=torch.nn.MSELoss(), optimizer=torch.optim.SGD, optimizer_parameters={"lr": 0.01}, n_epochs=10, verbose=True, validator=None):
    """
    Train an model for predicting input stream continuation.

    Parameters
    ----------
    model
        The model to be trained.
    data_loader
        Data loader used for training
    loss_function, default is torch.nn.MSELoss()
        Loss function to be optimized
    optimizer, default is torch.optim.SGD
        Optimizer to be used to minimize the loss function
    optimizer_parameters, default is {"lr": 0.01}
        Parameters for the optimizer
    n_epochs, default is 10
        Number of epochs (one epoch corresponds to iterating over all data)
    verbose, default is True
        Print an estimate of the training loss after every epoch to the console
    validator, default is None
        Validator object which can be used to validate the model after every epoch
    """

    optimizer = optimizer(model.parameters(), **optimizer_parameters)

    for epoch in range(n_epochs):
        epoch_loss = 0
        start_time = time.time()
        for batch in data_loader:
            # Reset gradients to zero
            optimizer.zero_grad()

            # Compute prediction error
            outputs = model(*batch[:-1])
            loss = loss_function(outputs, batch[-1])

            # Perform one step of backpropagation
            loss.backward()
            optimizer.step()

            # Accumulate the loss for this epoch
            epoch_loss += loss.item()

        if verbose:
            epoch_loss /= len(data_loader)
            msg = "[Epoch {:2d}/{}] \t".format(epoch + 1, n_epochs)
            msg += "Time: {:.2f} s - ".format(time.time() - start_time)
            msg += "Loss: {:.4f}".format(epoch_loss)

            if validator:
                msg += validator(model)

            print(msg)

class MLP(torch.nn.Sequential):
    """
    Multi-Layer Perceptron architecture: Fully connected neural network with
    dropout and activation function.
    """

    def __init__(self, layer_sizes, activation=torch.nn.ReLU(), output_activation=None, dropout=0.0):
        """
        Parameters
        ----------
        layer_sizes : list of int
            List of layer sizes. Format:
                [input_size, hidden_size_1, hidden_size_2, ..., output_size]
        activation : torch.nn.Module, optional, default is torch.nn.ReLU()
            Activation function between the fully connected layers.
        dropout : float, optional, default is 0.0
            Dropout probability in each of the layers.
        """
        layers = []
        in_dim = layer_sizes[0]
        for out_dim in layer_sizes[1:-1]:
            layers.append(torch.nn.Linear(in_dim, out_dim))
            layers.append(activation)
            layers.append(torch.nn.Dropout(dropout))
            in_dim = out_dim

        layers.append(torch.nn.Linear(in_dim, layer_sizes[-1]))
        if output_activation:
            layers.append(output_activation)
        super().__init__(*layers)

import torchvision

model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 1, kernel_size=(3, 3)),
    torch.nn.Softplus(),
    torch.nn.BatchNorm2d(1),
    torch.nn.Conv2d(1, 1, kernel_size=(3, 3)),
    torch.nn.Softplus(),
    torch.nn.BatchNorm2d(1),
    torch.nn.Flatten(),
    torch.nn.Linear(576, 512),
    torch.nn.Softplus(),
    torch.nn.Linear(512, 512),
    torch.nn.Softplus(),
    torch.nn.Linear(512, 10),
    torch.nn.Sigmoid(),
)

train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('../mnist_data', 
                                           download=True,
                                           train=True,
                                           transform=torchvision.transforms.ToTensor()),
                                           batch_size=64,
                                           shuffle=True
)

test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('../mnist_data', 
                                          download=True,
                                          train=False,
                                          transform=torchvision.transforms.ToTensor()),
                                          batch_size=64,
                                          shuffle=False
)

loss_function = lambda x, y: torch.nn.MSELoss()(x, torch.nn.functional.one_hot(y, num_classes=10).to(dtype=torch.float))

def validator(model):
    with torch.no_grad():
        hits = 0
        for batch in test_loader:
            hits += (torch.argmax(model(*batch[:-1]), dim=1) == batch[-1]).sum()
    accuracy = hits / len(test_loader.dataset)
    msg = " - Accuracy: {:.4f}".format(accuracy)
    return msg

train(model, train_loader, loss_function, validator=validator, n_epochs=1)

data = next(iter(train_loader))
H = hessian(model, loss_function, data, spectral_transform=True)

num_param = sum(p.numel() for p in model.parameters())
X = torch.randn(num_param, 10)
