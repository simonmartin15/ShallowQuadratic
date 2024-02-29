import matplotlib.pyplot as plt
import torch
import Model as md
from ClearCache import ClearCache


plt.style.use("bmh")
plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=17)
plt.rc('ytick', labelsize=17)
plt.rc('axes', unicode_minus=False)
plt.rc('legend', fontsize=17)
plt.rcParams["text.usetex"] = False


def main():
    """Plots high dimensional limit for the function phi in the orthonormal case"""

    print('')
    print('Figure 2')

    seed = 0
    rep = 5
    nstep = 10
    eta = 1e-2
    d = [20, 40, 80, 200]
    args = (True, True, False, True)

    print('Simulating (Finite Dimension)')
    print('m > m*')
    m1 = [10, 20, 40, 100]
    ms1 = [5, 10, 20, 50]
    with ClearCache():
        Sim1 = md.Simulator(d, ms1, m1, rep, args, seed, eta)
        Sim1.simulate_GD(nstep)

    print('m = m*')
    m2 = [10, 20, 40, 100]
    ms2 = [10, 20, 40, 100]
    with ClearCache():
        Sim2 = md.Simulator(d, ms2, m2, rep, args, seed, eta)
        Sim2.simulate_GD(nstep)

    print('m < m*')
    m3 = [5, 10, 20, 50]
    ms3 = [10, 20, 40, 100]
    with ClearCache():
        Sim3 = md.Simulator(d, ms3, m3, rep, args, seed, eta)
        Sim3.simulate_GD(nstep)

    print('Simulating (Infinite Dimension)')
    alpha = [0.5, 0.5, 0.25]
    alphastar = [0.25, 0.5, 0.5]
    eta_inf = 2e-5
    steps_inf = []
    Phi_inf = []
    Over_inf = []

    for i in range(3):
        Sim_inf = md.SimulatorImplicit(alpha[i], alphastar[i], over=True)
        Sim_inf.optimize(end=10, eta=eta_inf)
        steps_inf.append(Sim_inf.steps)
        Phi_inf.append(Sim_inf.Phi)
        Over_inf.append(Sim_inf.overlap)
    
    Phi1 = []
    Phi2 = []
    Phi3 = []
    Over1 = []
    Over2 = []
    Over3 = []

    for i in range(4):
        Phi1.append(torch.mean(torch.stack(Sim1.SumStats['PHI'][rep * i:rep * (i + 1)]), dim=0).detach().numpy())
        Phi2.append(torch.mean(torch.stack(Sim2.SumStats['PHI'][rep * i:rep * (i + 1)]), dim=0).detach().numpy())
        Phi3.append(torch.mean(torch.stack(Sim3.SumStats['PHI'][rep * i:rep * (i + 1)]), dim=0).detach().numpy())
        Over1.append(torch.mean(torch.stack(Sim1.SumStats['Overlap'][rep * i:rep * (i + 1)]), dim=0).detach().numpy())
        Over2.append(torch.mean(torch.stack(Sim2.SumStats['Overlap'][rep * i:rep * (i + 1)]), dim=0).detach().numpy())
        Over3.append(torch.mean(torch.stack(Sim3.SumStats['Overlap'][rep * i:rep * (i + 1)]), dim=0).detach().numpy())


    labels = ['$d=20$', '$d=40$', '$d=100$', '$d=200$']
    colors = ['tab:orange', 'tab:red', 'tab:green', 'tab:purple']

    steps = []
    for i in range(4):
        steps.append(Sim1.steps[i * rep].detach().numpy() * eta / Sim1.d[i])


    # Phi
    fig = plt.figure(figsize=(7, 10))
    ax1 = fig.add_subplot(3, 1, 1)

    for i in range(4):
        plt.plot(steps[i], Phi1[i], label=labels[i], color=colors[i], linewidth=2)

    plt.plot(steps_inf[0], Phi_inf[0], label=r'$d=\infty$', color='black', linestyle='dashed')

    plt.text(0.6, 0.83, r'$\alpha > \alpha^*$', fontsize=25,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'), transform=ax1.transAxes)

    plt.grid('both')
    plt.legend()
    plt.xscale('log')
    plt.xlim(1e-3, 10)
    plt.xlabel(r'$\gamma$', labelpad=5)
    plt.ylabel(r'$\phi_d(\gamma)$', rotation=0, labelpad=27)

    ax2 = fig.add_subplot(3, 1, 2)

    for i in range(4):
        plt.plot(steps[i], Phi2[i], label=labels[i], color=colors[i], linewidth=2)
    plt.plot(steps_inf[1], Phi_inf[1], label=r'$d=\infty$', color='black', linestyle='dashed')

    plt.text(0.6, 0.83, r'$\alpha = \alpha^*$', fontsize=25,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'), transform=ax2.transAxes)

    plt.grid('both')
    plt.xscale('log')
    plt.xlim(1e-3, 10)
    plt.xlabel(r'$\gamma$', labelpad=5)
    plt.ylabel(r'$\phi_d(\gamma)$', rotation=0, labelpad=27)

    ax3 = fig.add_subplot(3, 1, 3)

    for i in range(4):
        plt.plot(steps[i], Phi3[i], label=labels[i], color=colors[i], linewidth=2)
    plt.plot(steps_inf[2], Phi_inf[2], label=r'$d=\infty$', color='black', linestyle='dashed')

    plt.text(0.6, 0.83, r'$\alpha < \alpha^*$', fontsize=25,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'), transform=ax3.transAxes)

    plt.grid('both')
    plt.xlim(0, 10)
    plt.xlabel(r'$\gamma$', labelpad=5)
    plt.ylabel(r'$\phi_d(\gamma)$', rotation=0, labelpad=27)

    plt.tight_layout()
    plt.savefig('Figure2.pdf', format='pdf')


    print('')
    print('Figure 3')

    # Overlap
    fig = plt.figure(figsize=(7, 10))
    ax1 = fig.add_subplot(3, 1, 1)

    for i in range(4):
        plt.plot(steps[i], Over1[i], label=labels[i], color=colors[i], linewidth=2)

    plt.plot(steps_inf[0], Over_inf[0], label=r'$d=\infty$', color='black', linestyle='dashed')

    plt.text(0.2, 0.83, r'$\alpha > \alpha^*$', fontsize=25,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'), transform=ax1.transAxes)

    plt.grid('both')
    plt.legend(loc='lower right')
    plt.xscale('log')
    plt.xlim(1e-3, 10)
    plt.ylim(0.3, 1.05)
    plt.xlabel(r'$\gamma$', labelpad=5)
    plt.ylabel(r'$\chi_d(\gamma)$', rotation=0, labelpad=27)

    ax2 = fig.add_subplot(3, 1, 2)

    for i in range(4):
        plt.plot(steps[i], Over2[i], label=labels[i], color=colors[i], linewidth=2)
    plt.plot(steps_inf[1], Over_inf[1], label=r'$d=\infty$', color='black', linestyle='dashed')

    plt.text(0.2, 0.83, r'$\alpha = \alpha^*$', fontsize=25,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'), transform=ax2.transAxes)

    plt.grid('both')
    plt.xscale('log')
    plt.xlim(1e-3, 10)
    plt.ylim(0.45, 1.05)
    plt.xlabel(r'$\gamma$', labelpad=5)
    plt.ylabel(r'$\chi_d(\gamma)$', rotation=0, labelpad=27)

    ax3 = fig.add_subplot(3, 1, 3)

    for i in range(4):
        plt.plot(steps[i], Over3[i], label=labels[i], color=colors[i], linewidth=2)
    plt.plot(steps_inf[2], Over_inf[2], label=r'$d=\infty$', color='black', linestyle='dashed')

    plt.text(0.2, 0.83, r'$\alpha < \alpha^*$', fontsize=25,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'), transform=ax3.transAxes)

    plt.grid('both')
    plt.xscale('log')
    plt.xlim(1e-3, 10)
    plt.ylim(0.3, 0.9)
    plt.xlabel(r'$\gamma$', labelpad=5)
    plt.ylabel(r'$\chi_d(\gamma)$', rotation=0, labelpad=27)

    plt.tight_layout()
    plt.savefig('Figure3.pdf', format='pdf')


if __name__ == '__main__':
    main()
