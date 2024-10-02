# Randomized trace estimation for parameter-dependent matrices applied to spectral density approximation

![](https://img.shields.io/badge/-Compatibility-gray?style=flat-square) &ensp;
![](https://img.shields.io/badge/Python_3.8+-white?style=flat-square&logo=python&color=white&logoColor=white&labelColor=gray)

![](https://img.shields.io/badge/-Dependencies-gray?style=flat-square)&ensp;
![](https://img.shields.io/badge/NumPy-white?style=flat-square&logo=numpy&color=white&logoColor=white&labelColor=gray)
![](https://img.shields.io/badge/SciPy-white?style=flat-square&logo=scipy&color=white&logoColor=white&labelColor=gray)
![](https://img.shields.io/badge/Matplotlib-white?style=flat-square&logo=python&color=white&logoColor=white&labelColor=gray)
![](https://img.shields.io/badge/PyTorch-white?style=flat-square&logo=pytorch&color=white&logoColor=white&labelColor=gray)

## Quick start

### Prerequisites

To reproduce our results, you will need

- a [Git](https://git-scm.com/downloads) installation to clone the repository;
- a recent version of [Python](https://www.python.org/downloads) to run the experiments;

> [!NOTE]
> The commands `git` and `python` have to be discoverable by your terminal. To verify this, type `[command] --version` in your terminal.

### Setup

Clone this repository using
```[shell]
git clone https://github.com/FMatti/Rand-TRACE
cd Rand-TRACE
```

Install all the requirements with
```[shell]
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Reproduce the whole project with the command
```[shell]
python -m reproduce.py -a
```
> [!NOTE]
> Reproducing the whole project might take up to one hour!

### Tests

To run the tests, you will need to install [pytest](https://docs.pytest.org/en/stable/) and run the command `pytest` at the root of this project with the commands

```[shell]
python -m pip install pytest
pytest
```

## Theoretical background


We consider parameter-dependent matrices of the form

$$
    \boldsymbol{B}(t) = \begin{bmatrix}
        b_{11}(t) & b_{12}(t) & \dots & b_{1n}(t) \\
        b_{21}(t) & b_{22}(t) & \dots & b_{2n}(t) \\
        \vdots & \vdots & \ddots & \vdots \\
        b_{n1}(t) & b_{n2}(t) & \dots & b_{nn}(t) \\
    \end{bmatrix} \in \mathbb{R}^{n \times n}
$$

where $b_{ij}(t)$ are functions depending continuously on the parameter $t$ which takes values in the interval $[a,b]$. The trace of such a matrix is defined as

$$
    \mathrm{Tr}(\boldsymbol{B}(t)) = \sum_{i=1}^{n} b_{ii}(t).
$$

However, we assume that we only have access to products of this matrix with vectors for each $t \in [a, b]$, so this definition will not be directly useful for computing the trace.

### Girard-Hutchinson estimator

We can approximate the trace with the Girard-Hutchinson estimator: We take $n_{\boldsymbol{\Psi}}$ stochastically independent standard Gaussian random vectors $\boldsymbol{\psi}_1,\dots, \boldsymbol{\psi}_{n_{\boldsymbol{\Psi}}} \in \mathbb{R}^{n}$ to form

$$
    \mathrm{Tr}_{\boldsymbol{\Psi}}(\boldsymbol{B}(t))
    = \frac{1}{n_{\boldsymbol{\Psi}}} \sum_{j=1}^{n_{\boldsymbol{\Psi}}} \boldsymbol{\psi}_j^{\top} \boldsymbol{B}(t) \boldsymbol{\psi}_j
    = \frac{1}{n_{\boldsymbol{\Psi}}} \mathrm{Tr}( \boldsymbol{\Psi}^{\top} \boldsymbol{B}(t) \boldsymbol{\Psi})
$$

where $\boldsymbol{\Psi} = [\boldsymbol{\psi}_1 ~ \cdots ~ \boldsymbol{\psi}_{n_{\boldsymbol{\Psi}}}] \in \mathbb{R}^{n \times n_{\boldsymbol{\Psi}}}$. Other choices for the distribution of the random vectors are possible, for example by uniformly sampling from $\{-1, +1\}$ or from the $(n-1)$-sphere. However, our theoretical developments only hold in the Gaussian case.

### Nyström estimator

Alternatively, the trace of a symmetric matrix whose singular values decay quickly can be approximated well by using a Gaussian sketching matrix $\boldsymbol{\Omega} \in \mathbb{R}^{n \times n_{\boldsymbol{\Omega}}}$ to form the Nyström approximation

$$
    \boldsymbol{B}_{\boldsymbol{\Omega}}(t) = (\boldsymbol{B}(t) \boldsymbol{\Omega}) (\boldsymbol{\Omega}^{\top} \boldsymbol{B}(t) \boldsymbol{\Omega})^{\dagger} (\boldsymbol{B}(t) \boldsymbol{\Omega})^{\top}.
$$

Then we can estimate the trace as $\mathrm{Tr}(\boldsymbol{B}_{\boldsymbol{\Omega}}(t))$. Thanks to the invariance of the trace under cyclic permutation of its arguments and the symmetry of the matrix, we may rewrite this estimator as

$$
    \mathrm{Tr}(\boldsymbol{B}_{\boldsymbol{\Omega}}(t)) = \mathrm{Tr}( (\boldsymbol{\Omega}^{\top} \boldsymbol{B}(t) \boldsymbol{\Omega})^{\dagger} ( \boldsymbol{\Omega}^{\top} \boldsymbol{B}(t)^2 \boldsymbol{\Omega})).
$$

### Nyström++ estimator

Finally, an estimator which corrects for inaccuracies in the Nyström approximation by estimating the trace of its residual using the Girard-Hutchinson estimator is

$$
    \mathrm{Tr}_{\boldsymbol{\Psi}, \boldsymbol{\Omega}}(\boldsymbol{B}(t)) = \mathrm{Tr}(\boldsymbol{B}_{\boldsymbol{\Omega}}(t)) + \mathrm{Tr}_{\boldsymbol{\Psi}}(\boldsymbol{B}(t) - \boldsymbol{B}_{\boldsymbol{\Omega}}(t)).
$$

This is the parameter-dependent analogue of the Nyström++ estimator, which is based on the Hutch++ estimator.

## Project structure

```
Rand-TRACE
│   README.md           (file you are reading right now)
|   requirements.txt    (python package requirements file)
|   reproduce.py        (script for easy setup of project)
|
└───paper               (the LaTeX project for the paper)
└───reproduce           (scripts which help setup and reproduce project)
└───algorithms          (the algorithms introduced in the paper)
└───matrices            (the example matrices used for the numerical results)
└───test                (unit tests written for the algorithms)
```
