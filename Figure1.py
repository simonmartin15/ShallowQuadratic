import matplotlib.pyplot as plt
import numpy as np
import torch


plt.style.use("bmh")
plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=17)
plt.rc('ytick', labelsize=17)
plt.rc('axes', unicode_minus=False)
plt.rc('legend', fontsize=17)
plt.rcParams["text.usetex"] = True


def pdf(r1, r2, x):
    """Probability density function associated with the measure mu"""
    return np.sqrt((r2 - x) * (x - r1)) / (2 * np.pi * x * (1-x))


def sample_eigenvalues(alpha, alphastar, d):
    """Samples the ESD of the matrix Y_d"""
    m = int(alpha * d)
    ms = int(alphastar * d)

    Ws = torch.randn(size=(d, ms))
    Us = torch.svd(Ws)[0]  # Us.T @ Us = Id (sampled uniformly)

    W = torch.randn(size=(d, m))
    U = torch.svd(W)[0]  # U.T @ U = Id (sampled uniformly)

    Y = U.T @ Us @ Us.T @ U
    return torch.real(torch.linalg.eigvals(Y))


def main():
    alpha = 0.3
    alphastar = 0.5
    r1 = (np.sqrt(alpha * (1 - alphastar)) - np.sqrt(alphastar * (1 - alpha))) ** 2
    r2 = (np.sqrt(alpha * (1 - alphastar)) + np.sqrt(alphastar * (1 - alpha))) ** 2
    eps = 1e-6
    x = np.linspace(r1+eps, r2-eps, 1000)
    y = pdf(r1, r2, x) / alpha

    torch.manual_seed(seed=0)

    eig = sample_eigenvalues(alpha, alphastar, d=2000)
    counts, bins = np.histogram(eig, bins=30)

    print('')
    print('Figure 1')

    figure = plt.figure(figsize=(8, 4))

    plt.plot(x, y, color='black', linewidth=2, linestyle='dashed', label=r'$d = \infty$')
    plt.hist(bins[:-1], bins, weights=counts, color='lightsteelblue', ec='black', density=True, label=r'$d = 2000$')

    plt.xlabel(r'$x$', labelpad=-2)
    plt.ylabel(r'$\mathrm{PDF}(x)$', rotation=0, labelpad=40)
    plt.legend(loc='center', bbox_to_anchor=(0.5, 0.85))
    plt.xlim(0.02, 0.98)
    plt.ylim(0, 1.7)
    plt.tight_layout()
    figure.savefig('Figure1.pdf', format='pdf')


if __name__ == '__main__':
    main()
