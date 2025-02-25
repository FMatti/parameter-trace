# Stochastic trace estimation for parameter-dependent matrices applied to spectral density approximation

![](https://img.shields.io/badge/-Compatibility-gray?style=flat-square) &ensp;
![](https://img.shields.io/badge/Python\_3.8+-white?style=flat-square&logo=python&color=white&logoColor=white&labelColor=gray)

![](https://img.shields.io/badge/-Dependencies-gray?style=flat-square)&ensp;
![](https://img.shields.io/badge/NumPy-white?style=flat-square&logo=numpy&color=white&logoColor=white&labelColor=gray)
![](https://img.shields.io/badge/SciPy-white?style=flat-square&logo=scipy&color=white&logoColor=white&labelColor=gray)
![](https://img.shields.io/badge/Matplotlib-white?style=flat-square&logo=python&color=white&logoColor=white&labelColor=gray)
![](https://img.shields.io/badge/PyTorch-white?style=flat-square&logo=pytorch&color=white&logoColor=white&labelColor=gray)

## Abstract

Stochastic trace estimation is a well-established tool for approximating the trace of a large symmetric positive semi-definite matrix $\boldsymbol{B}$. Several applications involve a matrix that depends continuously on a parameter $t \in [a,b]$, and require trace estimates of $\boldsymbol{B}(t)$ for many different values of $t$. This is, for example, the case when approximating the spectral density of a matrix or computing partition functions of quantum systems. Approximating the trace separately for each matrix $\boldsymbol{B}(t_1), \dots, \boldsymbol{B}(t_m)$ clearly incurs redundancies and a cost that scales linearly with $m$. To address this issue, we propose and analyze modifications for three stochastic trace estimators, the Girard-Hutchinson, Nyström, and Nyström++ estimators. Our modification uses \emph{constant} randomization across different values of $t$, that is, every matrix $\boldsymbol{B}(t_1), \dots, \boldsymbol{B}(t_m)$ is multiplied with the \emph{same} set of random vectors. When combined with Chebyshev approximation in $t$, the use of such constant random matrices allows one to reuse matrix-vector products across different values of $t$, leading to significant cost reduction. Our analysis shows that the loss of stochastic independence across different $t$ does not lead to deterioration; in fact, our error bounds for these estimators closely match existing error bounds for the constant case. In particular, we show that $\mathcal{O}(\varepsilon^{-1})$ random matrix-vector products suffice to ensure an error of $\varepsilon > 0$ for Nyström++, independent of low-rank properties of $\boldsymbol{B}(t)$. We discuss in detail how the combination of Nyström++ with Chebyshev approximation applies to large-scale spectral density estimation and provide an analysis of the resulting method. This improves various aspects of an existing stochastic estimator for spectral density estimation. Several numerical experiments for examples from electronic structure interaction, statistical thermodynamics, and neural network optimization illustrate and validate our findings.

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
git clone https://github.com/FMatti/parameter-trace
cd parameter-trace
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
> Reproducing the whole project might around one hour!

### Tests

To run the tests written for the algorithms developed and used in the paper, you will need to install [pytest](https://docs.pytest.org/en/stable/) and run the command `pytest` at the root of this project with the commands

```[shell]
python -m pip install pytest
pytest
```

## Paper overview

We consider parameter-dependent matrices of the form

$$
    \boldsymbol{B}(t) = \begin{bmatrix}
        b\_{11}(t) & b\_{12}(t) & \dots & b\_{1n}(t) \\
        b\_{21}(t) & b\_{22}(t) & \dots & b\_{2n}(t) \\
        \vdots & \vdots & \ddots & \vdots \\
        b\_{n1}(t) & b\_{n2}(t) & \dots & b\_{nn}(t) \\
    \end{bmatrix} \in \mathbb{R}^{n \times n}
$$

where $b\_{ij}(t)$ are functions depending continuously on the parameter $t$ which takes values in the interval $[a,b]$. The trace of such a matrix is defined as

$$
    \mathrm{Tr}(\boldsymbol{B}(t)) = \sum\_{i=1}^{n} b\_{ii}(t).
$$

However, we assume that we only have access to products of this matrix with vectors for each $t \in [a, b]$, so this definition will not be directly useful for computing the trace.

### Girard-Hutchinson estimator

We can approximate the trace with the Girard-Hutchinson estimator: We take $n\_{\boldsymbol{\Psi}}$ stochastically independent Gaussian random vectors $\boldsymbol{\psi}\_1,\dots, \boldsymbol{\psi}\_{n\_{\boldsymbol{\Psi}}} \in \mathbb{R}^{n}$ to form

$$
    \mathrm{Tr}\_{\boldsymbol{\Psi}}(\boldsymbol{B}(t))
    = \frac{1}{n\_{\boldsymbol{\Psi}}} \sum\_{j=1}^{n\_{\boldsymbol{\Psi}}} \boldsymbol{\psi}\_j^{\top} \boldsymbol{B}(t) \boldsymbol{\psi}\_j
    = \frac{1}{n\_{\boldsymbol{\Psi}}} \mathrm{Tr}( \boldsymbol{\Psi}^{\top} \boldsymbol{B}(t) \boldsymbol{\Psi})
$$

where $\boldsymbol{\Psi} = [\boldsymbol{\psi}\_1 ~ \cdots ~ \boldsymbol{\psi}\_{n\_{\boldsymbol{\Psi}}}] \in \mathbb{R}^{n \times n\_{\boldsymbol{\Psi}}}$.

### Nyström estimator

Alternatively, the trace of a symmetric matrix whose singular values decay quickly can be approximated well by using a Gaussian sketching matrix $\boldsymbol{\Omega} \in \mathbb{R}^{n \times n\_{\boldsymbol{\Omega}}}$ to form the Nyström approximation

$$
    \widehat{\boldsymbol{B}}\_{\boldsymbol{\Omega}}(t) = (\boldsymbol{B}(t) \boldsymbol{\Omega}) (\boldsymbol{\Omega}^{\top} \boldsymbol{B}(t) \boldsymbol{\Omega})^{\dagger} (\boldsymbol{B}(t) \boldsymbol{\Omega})^{\top}.
$$

Then we can estimate the trace as $\mathrm{Tr}(\boldsymbol{B}\_{\boldsymbol{\Omega}}(t))$. Thanks to the invariance of the trace under cyclic permutation of its arguments and the symmetry of the matrix, we may rewrite this estimator as

$$
    \mathrm{Tr}(\boldsymbol{B}\_{\boldsymbol{\Omega}}(t)) = \mathrm{Tr}( (\boldsymbol{\Omega}^{\top} \boldsymbol{B}(t) \boldsymbol{\Omega})^{\dagger} ( \boldsymbol{\Omega}^{\top} \boldsymbol{B}(t)^2 \boldsymbol{\Omega})).
$$

### Nyström++ estimator

Finally, an estimator which corrects for inaccuracies in the Nyström approximation by estimating the trace of its residual using the Girard-Hutchinson estimator is

$$
    \mathrm{Tr}\_{\boldsymbol{\Psi}, \boldsymbol{\Omega}}(\boldsymbol{B}(t)) = \mathrm{Tr}(\widehat{\boldsymbol{B}}\_{\boldsymbol{\Omega}}(t)) + \mathrm{Tr}\_{\boldsymbol{\Psi}}(\boldsymbol{B}(t) - \widehat{\boldsymbol{B}}\_{\boldsymbol{\Omega}}(t)).
$$

This is the parameter-dependent analogue of the Nyström++ estimator, which is based on the Hutch++ estimator.

### Chebyshev-Nyström++ method

The smoothed spectral density of a real symmetric matrix $\boldsymbol{A} \in \mathbb{R}^{n \times n}$ with eigenvalues $\lambda\_1, \dots, \lambda\_n \in \mathbb{R}$ is defined as 

$$
    \phi_{\sigma}(t) = \frac{1}{n} \sum_{i=1} g\_{\sigma}(t - \lambda\_i)
$$

for a Gaussian $g\_{\sigma}$. Basic matrix function manipulatons show that

$$
    \phi_{\sigma}(t) = \frac{1}{n} \mathrm{Tr}(g\_{\sigma}(t \boldsymbol{I}\_n - \boldsymbol{A})),
$$

which allows us to apply any of the above presented stochastic trace estimators to the parameter dependent matrix $g\_{\sigma}(t \boldsymbol{I}\_n - \boldsymbol{A})$. To evaluate $g\_{\sigma}(t \boldsymbol{I}\_n - \boldsymbol{A})$, we approximate $g\_{\sigma}$ with its degree $m$ Chebyshev interpolant

$$
    g\_{\sigma}^{(m)}(t - s) = \sum\_{l=0}^{m} \mu\_{l}(t) T\_l(\boldsymbol{A}),
$$

whose coefficients $\mu\_{l}(t)$ we can compute through a discrete cosine transform.

Finally the Chebyshev-Nyström++ estimator approximates the smoothed spectral density by applying the Nyström++ approximation to the Chebyshev interpolant

$$
    \phi_{\sigma}(t) \approx \mathrm{Tr}\_{\boldsymbol{\Psi}, \boldsymbol{\Omega}}(g\_{\sigma}^{(m)}(t \boldsymbol{I}\_n - \boldsymbol{A})).
$$

## Code structure

```
parameter-trace
│   README.md           (file you are reading right now)
|   requirements.txt    (Python package requirements file)
|   reproduce.py        (script for easy reproduction of project)
|
└───paper               (the LaTeX project for the paper)
└───reproduce           (scripts which help setup and reproduce project)
└───algorithms          (the algorithms introduced in the paper)
└───matrices            (the example matrices used for the numerical results)
└───test                (unit tests written for the algorithms)
```
