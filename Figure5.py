import matplotlib.pyplot as plt
import numpy as np
import torch
import Model as md
from ClearCache import ClearCache

plt.style.use("bmh")
plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=17)
plt.rc('ytick', labelsize=17)
plt.rc('axes', unicode_minus=False)
plt.rc('legend', fontsize=17)
plt.rcParams["text.usetex"] = True


def main():
    """Plots renormalized non-exponential convergence rates for the flow"""

    print('')
    print('Figure 5')

    seed = 0
    rep = 5
    nstep = 12500
    eta = 2e-3
    d = [20, 40, 100]
    m = [10, 20, 50]

    print('Simulating m > m*')
    ms1 = [5, 10, 25]
    args1 = (False, True, False, False)
    with ClearCache():
        Sim1 = md.Simulator(d, ms1, m, rep, args1, seed, eta)
        Sim1.simulate_GD(nstep)

    print('Simulating m = m*')
    ms2 = [10, 20, 50]
    args2 = (False, False, False, False)
    with ClearCache():
        Sim2 = md.Simulator(d, ms2, m, rep, args2, seed, eta)
        Sim2.simulate_GD(nstep)


    Loss1 = []
    Loss2 = []

    for i in range(3):
        L1 = torch.mean(torch.stack(Sim1.SumStats['Loss'][rep * i:rep * (i + 1)]), dim=0).detach().numpy()
        L2 = torch.mean(torch.stack(Sim2.SumStats['Loss'][rep * i:rep * (i + 1)]), dim=0).detach().numpy()
        Loss1.append(L1)
        Loss2.append(L2)

    labels = ['$d=20$', '$d=40$', '$d=100$']
    colors = ['tab:orange', 'tab:red', 'tab:green']

    steps1 = Sim1.eta * Sim1.steps[0].detach().numpy()
    steps2 = Sim2.eta * Sim2.steps[0].detach().numpy()

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    for i in range(3):
        plt.plot(steps1, steps1 ** 2 * Loss1[i], label=labels[i], color=colors[i], linewidth=2)

    plt.text(0.12, 0.15, r'$m > m^*$', fontsize=25, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'),
             transform=ax1.transAxes)

    plt.grid('both')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-2, steps1[-1])
    plt.ylim(2e-3, 2)
    plt.xlabel(r'$t$', labelpad=5)
    plt.ylabel(r'$\overline{\mathcal{L}}(t)$', rotation=0, labelpad=20)

    ax2 = fig.add_subplot(1, 2, 2)
    cst = [80, 5, 100]
    ind = [6000, 6000, 7500]

    for i in range(3):
        mu = Sim2.teacher[i][0][m[i] - 1, m[i] - 1]
        steps_bound = steps2[ind[i]:]
        plt.plot(steps2, Loss2[i], label=labels[i], color=colors[i], linewidth=2)
        plt.plot(steps_bound, cst[i] * np.exp(-4 * mu * steps_bound), color=colors[i], linestyle='dashed', alpha=0.7)

    plt.text(0.12, 0.15, r'$m = m^*$', fontsize=25, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'),
             transform=ax2.transAxes)

    plt.grid('both')
    plt.yscale('log')
    plt.xlim(0, 25)
    plt.ylim(1e-9, 1e2)
    plt.xlabel(r'$t$', labelpad=5)
    plt.ylabel(r'$\mathcal{L}(t)$', rotation=0, labelpad=20)

    plt.tight_layout()
    plt.savefig('Figure5.pdf', format='pdf')


if __name__ == '__main__':
    main()
