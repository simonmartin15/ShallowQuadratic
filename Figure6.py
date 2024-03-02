import matplotlib.pyplot as plt
import numpy as np
import Model as md

plt.style.use("bmh")
plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=17)
plt.rc('ytick', labelsize=17)
plt.rc('axes', unicode_minus=False)
plt.rc('legend', fontsize=17)

def main():
    end = 3
    eta = 2e-5
    a = np.array([0.5, 0.5, 0.25])
    astar = np.array([0.25, 0.5, 0.5])

    lims = np.minimum(1, np.sqrt(a/astar))

    print('')
    print('Figure 6')

    Y = []
    overlap = []

    print('Simulating')
    for i in range(3):
        Sim = md.SimulatorImplicit(a[i], astar[i], over=True)
        Sim.optimize(end, eta)
        Y.append(Sim.steps)
        overlap.append(Sim.overlap)
        

    labels = [r'$\alpha > \alpha^*$', r'$\alpha = \alpha^*$', r'$\alpha < \alpha^*$']
    colors = ['tab:orange', 'tab:red', 'tab:green']

    ind = 700
    Y_bound = [Y[0][ind:], Y[1][ind:], Y[2][ind:]]
    bounds = [Y_bound[0] ** (-2), np.sqrt(Y_bound[1]) * np.exp(-2 * Y_bound[1] / astar[1]),
              np.exp(-4 * Y_bound[2] / astar[2])]
    cst = [2e-3, 2, 0.5]

    plt.figure(figsize=(7, 4))
    for i in range(3):
        plt.plot(Y_bound[i], cst[i] * bounds[i], linestyle='dashed', color=colors[i], alpha=0.7)
    for i in range(3):
        delta = lims[i] - overlap[i]
        plt.plot(Y[i], delta, label=labels[i], color=colors[i])

    plt.grid('both')
    plt.legend()
    plt.yscale('log')
    plt.xlim(0, 3)
    plt.ylim(1e-7, 5)
    plt.ylabel(r'$\delta(\gamma)$', rotation=0, labelpad=20)
    plt.xlabel(r'$\gamma$', labelpad=5)

    plt.tight_layout()
    plt.savefig('Figure6.pdf', format='pdf')


if __name__ == '__main__':
    main()
