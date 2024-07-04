"""
hessian_density.py
------------------

Spectral density of Hessian matrix.
"""

import torch

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
    def __init__(self, model, loss_function, data):
        self.parameters = list(model.parameters())
        self.jacobian = compute_jacobian(model, loss_function, data)

    def __matmul__(self, X):
        hvp = hessian_vector_product(self.jacobian, self.parameters, X.T)
        hvp = torch.cat([e.reshape(X.shape[-1], -1) for e in hvp], dim=1).T
        return hvp

import time
import torch

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
        for i, (X, y) in enumerate(data_loader):
            # Reset gradients to zero
            optimizer.zero_grad()

            # Compute prediction error
            outputs = model(X)
            loss = loss_function(outputs, torch.nn.functional.one_hot(y, 10).to(dtype=torch.float))

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
                msg += validator.validate(model)

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

    def forward(self, x):
        return super().forward(x.view(-1, 28 * 28))

model = MLP([784, 512, 512, 10], activation=torch.nn.Sigmoid(), dropout=0.0)