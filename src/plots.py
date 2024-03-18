"""
Plots
-----

Helper functions for quickly creating convergence plots.
"""

from copy import deepcopy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from src.utils import spectral_transformation, form_spectral_density
from src.metrics import p_norm
from src.kernel import gaussian_kernel


def compute_spectral_densities(A, methods, labels, parameters, kernel=gaussian_kernel, n_t=1000, t=None, add_baseline=False):
    """
    Compute the spectral densities of a matrix A using different methods or parameters.

    Parameters
    ----------
    A : np.array or sp.sparse.matrix
        The matrix.
    methods : function or list of functions
        The method(s) used to compute the spectral densities.
    labels : function or list of functions
        The labels used as names for the computed spectral densities.
    parameters : function or list of functions
        The parameter(s) used to compute the spectral densities.
    kernel : function
        The smoothing kernel.
    n_t : int > 0
        The number of parameter values at which the spectral density is evaluated.
    t : np.ndarray (n_t,)
        (Optional) specific parameter values at which the spectral density is evaluated.
    add_baseline : bool
        Additionally compute the baseline of the spectral density.
    
    Returns
    -------
    spectral_densities : dict of np.ndarray
        The computed spectral densities for each of the methods/parameters.
    """
    parameters = deepcopy(parameters)

    eigenvalues = np.linalg.eigvalsh(A if isinstance(A, np.ndarray) else A.toarray())
    min_ev = np.min(eigenvalues)
    max_ev = np.max(eigenvalues)
    A_transformed = spectral_transformation(A, min_ev, max_ev)
    eigenvalues_transformed = spectral_transformation(eigenvalues, np.min(eigenvalues), np.max(eigenvalues))

    if not isinstance(methods, list):
        methods = [methods]
    if not isinstance(parameters, list):
        parameters = [parameters]
    if not isinstance(labels, list):
        if labels is None:
            labels = methods.copy()
        else:
            labels = [labels]

    if t is None:
        t = np.linspace(-1, 1, n_t)
    for parameter in parameters:
        parameter["sigma"] /= (max_ev - min_ev) / 2

    l = max(len(methods), len(parameters), len(labels))
    methods *= l if len(methods) < l else 1
    parameters *= l if len(parameters) < l else 1
    labels *= l if len(labels) < l else 1

    spectral_densities = {}
    if add_baseline:
        spectral_density_baseline = form_spectral_density(eigenvalues_transformed, kernel=kernel, n=A_transformed.shape[0], grid_points=t, sigma=parameters[0]["sigma"])
        spectral_densities["baseline"] = spectral_density_baseline

    for method, parameter, label in zip(methods, parameters, labels):
        spectral_density = method(A=A_transformed, t=t, **parameter)
        spectral_densities[label] = spectral_density

    return spectral_densities


def plot_spectral_densities(spectral_densities, parameters, variable_parameter=None, ignored_parameters=[], ax=None, colors=None, t=None):
    """
    Plot the spectral densities of a matrix A using different methods or parameters.

    Parameters
    ----------
    spectral_densities : dict of np.ndarray
        The computed spectral densities for each of the methods/parameters.
    parameters : function or list of functions
        The parameter(s) used to compute the spectral densities.
    variable_parameter : str
        The name of the variable parameter.
    ignored_parameters : list of str
        The names of the ignored parameters.
    kernel : function
        The smoothing kernel.
    ax : matplotlib.axes.Axes
        The axis on which the spectral density is plotted.
    colors : list of colors
        A list with the colors of the lines in the plot.
    t : np.ndarray (n_t,)
        (Optional) specific parameter values at which the spectral density is evaluated.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis on which the spectral density was plotted.
    """
    title = ""
    if isinstance(parameters, list):
        parameters = parameters[0]
    for key, value in parameters.items():
        if key == variable_parameter or key in ignored_parameters:
            continue
        title += "${}".format(key) if len(key) < 4 else"$\{}".format(key)
        title += " = {}$, ".format(value)
    title = title[:-2]
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    ax.set_title(title)
    ax.set_xlim([-1, 1])
    ax.set_ylabel("$\phi_{\sigma}$")
    ax.set_xlabel("$t$")

    if colors is None:
        colors = [matplotlib.colormaps["magma"](i / len(spectral_densities)) for i in range(len(spectral_densities))]

    for i, (label, spectral_density) in enumerate(spectral_densities.items()):
        if t is None:
            t = np.linspace(-1, 1, len(spectral_density))
        plt.plot(t, spectral_density, linewidth=1, color=colors[i], label=label)
    plt.legend()
    return ax


def compute_spectral_density_errors(A, methods, labels, variable_parameter, variable_parameter_values, parameters, kernel=gaussian_kernel, n_t=1000, error_metric=p_norm, correlated_parameter=None, correlated_parameter_values=None, eigenvalues=None):
    """
    Compute the errors of the spectral densities of a matrix A using different
    methods or parameters.

    Parameters
    ----------
    A : np.array or sp.sparse.matrix
        The matrix.
    methods : function or list of functions
        The method(s) used to compute the spectral densities.
    labels : function or list of functions
        The labels used as names for the computed spectral densities.
    variable_parameter : str
        The name of the variable parameter.
    variable_parameter_values : np.ndarray
        The values of the variable parameter.
    parameters : function or list of functions
        The parameter(s) used to compute the spectral densities.
    kernel : function
        The smoothing kernel.
    n_t : int > 0
        The number of parameter values at which the spectral density is evaluated.
    correlated_parameter : str
        A parameter which is correlated with the variable parameter.
    correlated_parameter_values : function
        The values which the correlated parameter depending on the variable parameter.
    eigenvalues : np.ndarray
        Eigenvalues which are used to form the baseline spectral density.
    
    Returns
    -------
    spectral_density_errors : dict of np.ndarray
        The errors for each of the methods/parameters depending on the variable parameter.
    """
    parameters = deepcopy(parameters)

    # Spectral transform of matrix
    if eigenvalues is None:
        eigenvalues = np.linalg.eigvalsh(A if isinstance(A, np.ndarray) else A.toarray())
    min_ev = np.min(eigenvalues)
    max_ev = np.max(eigenvalues)
    A_transformed = spectral_transformation(A, min_ev, max_ev)
    eigenvalues_transformed = spectral_transformation(eigenvalues, min_ev, max_ev)

    if not isinstance(methods, list):
        methods = [methods]
    if not isinstance(parameters, list):
        parameters = [parameters]
    if not isinstance(labels, list):
        if labels is None:
            labels = methods.copy()
        else:
            labels = [labels]

    for parameter in parameters:
        try:
            parameter["sigma"] /= (max_ev - min_ev) / 2
        except:
            pass

    l = max(len(methods), len(parameters), len(labels))
    methods *= l if len(methods) < l else 1
    parameters *= l if len(parameters) < l else 1
    labels *= l if len(labels) < l else 1

    t = np.linspace(-1, 1, n_t)

    spectral_density_errors = {}
    for label, method, parameter in zip(labels, methods, parameters):
        spectral_density_errors[label] = []
        for param in variable_parameter_values:
            if correlated_parameter is not None:
                parameter[correlated_parameter] = correlated_parameter_values(param)
                if correlated_parameter == "sigma":
                    parameter[correlated_parameter] /= (max_ev - min_ev) / 2
            if variable_parameter == "sigma":
                param /= (max_ev - min_ev) / 2
            try:
                kernel = parameter["kernel"]
            except:
                pass
            parameter[variable_parameter] = param
            spectral_density_baseline = form_spectral_density(eigenvalues_transformed, kernel=kernel, n=A_transformed.shape[0], n_t=n_t, sigma=parameter["sigma"])
            spectral_density = method(A=A_transformed, t=t, **parameter)
            spectral_density_errors[label].append(error_metric(spectral_density_baseline, spectral_density))

    return spectral_density_errors


def plot_spectral_density_errors(spectral_density_errors, parameters, variable_parameter, variable_parameter_values, error_metric_name="Error", x_label=None, ignored_parameters=[], ax=None, colors=None, markers=None, linestyles=None):
    """
    Plot the errors of the spectral densities of a matrix A using different
    methods or parameters.

    Parameters
    ----------
    spectral_density_errors : dict of np.ndarray
        The errors of the spectral densities for each of the methods/parameters.
    parameters : function or list of functions
        The parameter(s) used to compute the spectral densities.
    variable_parameter : str
        The name of the variable parameter.
    variable_parameter_values : np.ndarray
        The values of the variable parameter.
    error_metric_name : str
        The name of the error metric (y-label).
    x_label : str
        The name of the x-label.
    ignored_parameters : list of str
        The names of the ignored parameters.
    ax : matplotlib.axes.Axes
        The axis on which the spectral density is plotted.
    colors : list of colors
        A list with the colors of the lines in the plot.
    markers : list of markers
        A list with the types of markers used in the plot.
    linestyles : list of linestyles
        A list with the styles of the lines in the plot.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis on which the spectral density was plotted.
    """
    title = ""
    if isinstance(parameters, list):
        parameters = parameters[0]
    for key, value in parameters.items():
        if key == variable_parameter or key in ignored_parameters:
            continue
        title += "${}".format(key) if len(key) < 4 else"$\\{}".format(key)
        title += " = {}$, ".format(value)
    title = title[:-2]
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("{}".format(error_metric_name))
    ax.set_xlabel("${}$".format(variable_parameter) if x_label is None else x_label)

    if colors is None:
        colors = [matplotlib.colormaps["magma"](i / len(spectral_density_errors)) for i in range(len(spectral_density_errors))]
    if markers is None:
        markers = len(spectral_density_errors) * ["."]
    if linestyles is None:
        linestyles = len(spectral_density_errors) * ["-"]

    for i, (label, spectral_density_error) in enumerate(spectral_density_errors.items()):
        ax.plot(variable_parameter_values, spectral_density_error, linewidth=1, color=colors[i], label=label, marker=markers[i], linestyle=linestyles[i])
    ax.legend()
    return ax


# --- Unused implementations ---


def _plot_spectral_density_errors_heatmap(spectral_density_errors, variable_parameters, parameters, ignored_parameters=[], ax=None):
    title = ""
    if isinstance(parameters, list):
        parameters = parameters[0]
    for key, value in parameters.items():
        if key in variable_parameters.keys() or key in ignored_parameters:
            continue
        title += "${}".format(key) if len(key) < 4 else"$\{}".format(key)
        title += " = {}$, ".format(value)
    title = title[:-2]
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    param_1, param_2 = variable_parameters.keys()
    values_1, values_2 = variable_parameters.values()

    ax.set_title(title)
    ax.set_ylabel("${}$".format(param_1))
    ax.set_xlabel("${}$".format(param_2))
    ax.set_yticks(range(len(values_1)), values_1)
    ax.set_xticks(range(len(values_2)), values_2)

    plt.imshow(spectral_density_errors, cmap="magma", norm=matplotlib.colors.LogNorm())
    plt.colorbar()

    return ax


def _compute_spectral_density_errors_heatmap(A, method, variable_parameters, parameters, kernel=gaussian_kernel, n_t=1000, error_metric=p_norm):
    parameters = deepcopy(parameters)

    # Spectral transform of matrix
    eigenvalues = np.linalg.eigvalsh(A if isinstance(A, np.ndarray) else A.toarray())
    min_ev = np.min(eigenvalues)
    max_ev = np.max(eigenvalues)
    A_transformed = spectral_transformation(A, min_ev, max_ev)
    eigenvalues_transformed = spectral_transformation(eigenvalues, min_ev, max_ev)

    try:
        parameters["sigma"] /= (max_ev - min_ev) / 2
    except:
        pass

    param_1, param_2 = variable_parameters.keys()
    values_1, values_2 = variable_parameters.values()

    dos_errors = np.empty((len(values_1), len(values_2)))

    t = np.linspace(-1, 1, n_t)
    spectral_density_baseline = form_spectral_density(eigenvalues_transformed, kernel=kernel, n=A_transformed.shape[0], n_t=n_t, sigma=parameters["sigma"])
    for i, value_1 in enumerate(values_1):
        for j, value_2 in enumerate(values_2):
            spectral_density = method(A=A_transformed, t=t, **{param_1: value_1, param_2: value_2}, **parameters)
            dos_errors[i, j] = error_metric(spectral_density_baseline, spectral_density)

    return dos_errors
