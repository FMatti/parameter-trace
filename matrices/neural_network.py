"""
neural_network.py
-----------------

Spectral density of Hessian matrix.
"""

import torch
import time
import numpy as np
from algorithms.krylov_aware import lanczos


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


def train(model, data_loader, loss_function=torch.nn.MSELoss(), optimizer=torch.optim.SGD, optimizer_parameters={"lr": 0.01}, n_epochs=10, verbose=True, validator=None):
    """
    Train an model for predicting input stream continuation.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    data_loader
        Data loader used for training
    loss_function : torch.nn.Module
        Loss function to be optimized
    optimizer : torch.nn.Module
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
    loss_list = []

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

        epoch_loss /= len(data_loader)
        loss_list.append(epoch_loss)

        if verbose:
            msg = "[Epoch {:2d}/{}] \t".format(epoch + 1, n_epochs)
            msg += "Time: {:.2f} s - ".format(time.time() - start_time)
            msg += "Loss: {:.4f}".format(epoch_loss)

            if validator:
                msg += validator(model)

            print(msg)

    return loss_list


def accuracy_validator(model, data_loader):
    """
    Parameters
    ----------
    model : torch.nn.Module
        The model which is validated
    data_loader : torch.data.DataLoader
        Data on which the model will be validated.
    
    Returns
    ------
    msg : str
        The message which the validator prints to the console.
    """
    with torch.no_grad():
        hits = 0
        for batch in data_loader:
            hits += (torch.argmax(model(*batch[:-1]), dim=1) == batch[-1]).sum()
    accuracy = hits / len(data_loader.dataset)
    msg = " - Accuracy: {:.4f}".format(accuracy)
    return msg


def compute_gradient(model, loss_function, data):
    """
    Compute the gradient of each parameter for the model loss on some data.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network.
    loss_function : torch.nn.Module
        The loss function.
    data : tuple of torch.Tensor
        A tuple containing the inputs and targets of the neural network.

    Returns
    -------
    gradient : torch.Tensor
        The gradient of the model with respect to its parameters.
    """
    # Compute the loss of the model on some data
    input, target = data
    output = model(input)
    loss = loss_function(output, target)

    # Compute the gradient of the loss with respect to the model parameters
    gradient = torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
    gradient = torch.cat([grad.ravel() for grad in gradient])
    return gradient


def hessian_transpose_product(gradient, params, x):
    """
    Compute the product of the transpose of the Hessian with the vector x [1].

    Parameters
    ----------
    gradient : torch.Tensor
        The gradient of the model with respect to its parameters.
    params : list of torch.Tensor
        The model parameters for each layer of the neural network in a list.
    x : torch.Tensor
        The vector or matrix which the Hessian is multiplied with.

    Returns
    -------
    htp : torch.Tensor
        The result of the product of the Hessian transposed with x.

    References
    ----------
    [1] TODO
    """
    htp = torch.autograd.grad(gradient, params, x.T, retain_graph=True, is_grads_batched=x.ndim > 1)
    htp = torch.cat([e.reshape(x.shape[-1], -1) for e in htp], dim=1)
    return htp.T


class hessian(object):
    """
    Object which implements the product of the transpose of the Hessian matrix
    of a neural network with a vector or a matrix. Usage: H @ x

    Parameters
    ----------
    model : torch.nn.Module
        The neural network for which the Hessian is computed.
    loss_function : torch.nn.Module
        Loss function for which the Hessian is computed.
    data : torch.data.DataLoader
        Data loader for which the Hessian is computed.
    spectral_transform : bool
        Whether to normalize the Hessian such that its spectrum is in [-1, 1].
    """
    def __init__(self, model, loss_function, data, spectral_transform=True):
        # Determine the dimensions of the model
        self.parameters = list(model.parameters())
        self.num_parameters = sum(p.numel() for p in model.parameters())
        self.shape = (self.num_parameters, self.num_parameters)

        # Compute the gradient of the model
        self.gradient = compute_gradient(model, loss_function, data)

        # Normalize the spectrum of the Hessian
        self.scaling_parameter = 1.0
        if spectral_transform:
            self.scaling_parameter = self._approximate_spectral_norm()

    def _approximate_spectral_norm(self, k=10):
        """
        Approximate the spectral norm of the Hessian matrix [2].

        Parameters
        ----------
        k : int
            Number of Lanczos iterations.
        
        Returns
        -------
        spectral_norm : float
            Approximation of spectral norm.

        References
        ----------
        [2] TODO
        """
        x = torch.randn(self.num_parameters)
        Q, T = lanczos(self, x, k, dtype=np.float64)
        return np.max(np.linalg.eigvalsh(T[0, :-1, :])) + np.linalg.norm(T[0, -1, -1] * Q[:, :, -1])

    def update(self, model, loss_function, data):
        self.gradient = compute_gradient(model, loss_function, data)

    def __matmul__(self, x):
        # Convert numpy array to torch tensor
        is_numpy = isinstance(x, np.ndarray)
        if is_numpy:
            x = torch.from_numpy(x)

        # Compute the product of the Hessian transpose with x
        htp = hessian_transpose_product(self.gradient, self.parameters, x)

        # Rescale the result (for normalized Hessian)
        htp /= self.scaling_parameter

        # Convert back to numpy array
        if is_numpy:
            htp = htp.numpy()
        return htp
