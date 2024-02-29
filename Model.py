import torch
import torch.linalg as lin
import numpy as np


class Simulator:
    def __init__(self, d, mstar, m, rep, bools, seed, eta):

        torch.manual_seed(seed)

        self.eta = eta
        self.rep = rep  # number of repetitions of each simulation

        self.ortho, self.regenerate, self.bad_init, self.normalize = bools
        # regenerate = False => same teachers for each repetition (used for convergence rates only)

        self.generate = self.generate_ortho if self.ortho else self.generate_gaussian

        self.params = [d, mstar, m]
        self.Nrun, self.d, self.ms, self.m = self.reshape_parameters()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.teacher, self.init = self.generate()

        self.SumStats = None
        self.steps = None

    def reshape_parameters(self):
        """Reshape parameters of the simulator so that they have same shape"""
        d0 = self.params[0]
        ms0 = self.params[1]
        m0 = self.params[2]

        if len(torch.tensor(d0).shape) == 0:
            d = torch.tensor([d0])
            unique_d = True
        else:
            d = torch.tensor(d0)
            unique_d = False

        if len(torch.tensor(ms0).shape) == 0:
            ms = torch.tensor([ms0])
            unique_ms = True
        else:
            ms = torch.tensor(ms0)
            unique_ms = False

        if len(torch.tensor(m0).shape) == 0:
            m = torch.tensor([m0])
            unique_m = True
        else:
            m = torch.tensor(m0)
            unique_m = False

        Nrun = max(len(d), len(ms), len(m))
        if unique_d:
            d = torch.tile(d, (Nrun,))
        if unique_ms:
            ms = torch.tile(ms, (Nrun,))
        if unique_m:
            m = torch.tile(m, (Nrun,))

        return Nrun, d, ms, m

    def generate_gaussian(self):
        """Generates random teachers and initialization
        Teachers : list of diagonal d x d squared
        with diagonal distributed as eigenvalues of WW^T where W has N(0,Id/d) coefficients
        Initialization : list of d x m gaussian N(0, Id/d) matrices
        """
        Teacher = []
        Init = []

        r = self.rep if self.regenerate else 1

        for j in range(self.Nrun):
            teacher = torch.zeros(size=(r, self.d[j], self.d[j]))
            init = torch.zeros(size=(self.rep, self.d[j], self.m[j]))

            fac = 1 / np.sqrt(self.d[j]) if self.normalize else 1

            # teachers
            for k in range(r):
                Ws = fac * torch.randn(size=(self.d[j], self.ms[j])) / np.sqrt(self.ms[j])
                eigvals = torch.sort(torch.real(lin.eigvals(Ws.T @ Ws)), descending=True)[0]
                teacher[k] = torch.real(torch.diag(torch.cat((eigvals, torch.zeros((self.d[j] - self.ms[j], ))))))

            if not self.regenerate:
                Teacher.append(torch.tile(teacher[0][None, :], (self.rep, 1, 1)))
            else:
                Teacher.append(teacher)

            # initialization
            for k in range(self.rep):
                W = fac * torch.randn(size=(self.d[j], self.m[j])) / np.sqrt(self.m[j])
                if self.bad_init:
                    W[0] = torch.zeros((self.m[j], ))
                init[k] = W

            Init.append(init)

        return Teacher, Init

    def generate_ortho(self):
        """Generates random teachers and initialization
        Teachers : diagonal d x d orthogonal projection matrices
        Initialization : list of d x m orthogonal projection matrices
        """
        Teacher = []
        Init = []

        r = self.rep if self.regenerate else 1

        for j in range(self.Nrun):
            teacher = torch.zeros(size=(r, self.d[j], self.d[j]))
            init = torch.zeros(size=(self.rep, self.d[j], self.m[j]))

            # teachers
            for k in range(r):
                Ws = torch.randn(size=(self.d[j], self.ms[j]))
                Us = torch.svd(Ws)[0]
                teacher[k] = Us @ Us.T / self.ms[j]

            if not self.regenerate:
                Teacher.append(torch.tile(teacher[0][None, :], (self.rep, 1, 1)))
            else:
                Teacher.append(teacher)

            # initialization
            for k in range(self.rep):
                W = torch.randn(size=(self.d[j], self.m[j]))
                U = torch.svd(W)[0] / np.sqrt(self.m[j])
                if self.bad_init:
                    U[0] = torch.zeros((self.m[j], ))
                init[k] = U

            Init.append(init)

        return Teacher, Init

    def simulate_GD(self, nstep):
        """Gradient descent on population loss with no storing of the weights"""

        if nstep < 100:  # used to simulate high-dimensional limit
            nsteps = [int(np.ceil(nstep * d / self.eta)) for d in self.d]
        else:
            nsteps = [int(np.ceil(nstep)) for _ in self.d]

        DIST = []  # distance to teacher
        LOSS = []
        PHI = []  # psi(t) - tr(Z*)t (high-dimensional limit)
        OVERLAP = []

        steps = []

        for i in range(self.Nrun):
            for r in range(self.rep):
                print('\rRun [{0}/{1}], Rep [{2}/{3}], d = {4}, m* = {5}, m = {6}'.format(i + 1, self.Nrun, r + 1,
                                                                                          self.rep, self.d[i],
                                                                                          self.ms[i], self.m[i]),
                      end="")
                print('')

                Dist = torch.zeros(size=(nsteps[i], ))
                Loss = torch.zeros(size=(nsteps[i], ))
                Phi = torch.zeros(size=(nsteps[i], ))
                Overlap = torch.zeros(size=(nsteps[i], ))

                W = torch.clone(self.init[i][r].to(self.device))
                W.requires_grad_()

                teacher = self.teacher[i][r].to(self.device)
                phi = 0

                for j in range(nsteps[i]):
                    if (j+1) % 50 == 0:
                        print('\rStep [{0}/{1}]'.format(j + 1, nsteps[i]), end="")

                    Mat = W @ W.T
                    norm = lin.norm(Mat - teacher, ord='fro')
                    trace = torch.trace(Mat - teacher)

                    loss = norm ** 2 / 2 + trace ** 2 / 4

                    with torch.no_grad():
                        phi += self.eta * trace
                        overlap = torch.abs(torch.trace(Mat @ teacher) /
                                            (lin.norm(Mat, ord='fro') * lin.norm(teacher, ord='fro')))

                    Dist[j] = norm.item()
                    Loss[j] = loss.item()
                    Phi[j] = phi
                    Overlap[j] = overlap.item()

                    loss.backward()

                    with torch.no_grad():
                        W -= self.eta * W.grad

                    W.grad.zero_()

                print('')
                steps.append(torch.arange(nsteps[i]))

                DIST.append(Dist)
                LOSS.append(Loss)
                PHI.append(Phi)
                OVERLAP.append(Overlap)

        self.SumStats = {'Dist': DIST, 'Loss': LOSS, 'PHI': PHI, 'Overlap': OVERLAP}
        self.steps = steps.copy()


class SimulatorImplicit:
    """Simulate implicit equation for phi in the high-dimensional limit
    Also simulates the high-dimensional limit for the overlap
    """
    def __init__(self, a, astar, over):

        self.a = a
        self.astar = astar

        self.r1 = (np.sqrt(self.a * (1 - self.astar)) - np.sqrt(self.astar * (1 - self.a))) ** 2
        self.r2 = (np.sqrt(self.a * (1 - self.astar)) + np.sqrt(self.astar * (1 - self.a))) ** 2

        self.eta = 1e-5

        self.over = over  # indicates if we need to compute the overlap

        if self.a == self.astar and self.over:
            if self.a == 0.5:  # overlap computation is unstable when alpha = alpha*.
                # However, explicit solution in the case where alpha = alpha* = 1/2
                self.compute_overlap = self.compute_overlap_formula
            else:
                raise ValueError('Unable to compute the overlap. alpha = alpha* != 1/2 not implemented yet.')
        else:
            self.compute_overlap = self.compute_overlap_integrate

        self.overlap = None
        self.steps = None
        self.Phi = None

    def ThetaPrime(self, x):
        """Derivative of the function Theta from Proposition 5.1"""
        a1 = np.sqrt(1 + x * self.r1)
        a2 = np.sqrt(1 + x * self.r2)
        b1 = np.sqrt(self.r1)
        b2 = np.sqrt(self.r2)
        c1 = np.sqrt(1 - self.r1)
        c2 = np.sqrt(1 - self.r2)

        t1 = (self.r1 / a1 + self.r2 / a2) / (a1 + a2)
        t2 = - self.r1 * self.r2 * (b1 / a1 + b2 / a2) / (b2 * a1 + b1 * a2)
        t3 = - c1*c2 * (self.r1 * c2 / a1 + self.r2 * c1 / a2) / (c2 * a1 + c1 * a2)

        return (t1 + t2 + t3) / 2

    def func(self, y, j, f):
        """Computes phi from F, J = G/F
        Obtained by differentiating equation (12)"""
        t = (np.exp(4*y / self.astar) - j) * self.ThetaPrime(j-1) / self.a
        if self.astar + self.a > 1:
            t += (self.a + self.astar - 1) / self.a * (np.exp(4*y / self.astar) / j - 1)
        return - np.log(f) / 2 + np.log(1+t) / 2

    def omega1(self, x, e, J):
        """Integrating omega1 wrt x gives the numerator of the overlap (A(gamma) in Lemma C.11)"""
        # return np.sqrt((self.r2 - x) * (x - self.r1)) / (2 * np.pi * (1-x) * (1 + (J-1) * x))
        return e * np.sqrt((self.r2 - x) * (x - self.r1)) / (2 * np.pi * (1-x) * (1 + (J-1) * x))


    def omega2(self, x, e, J):
        """Integrating omega1 wrt x gives the denominator of the overlap (B(gamma) in Lemma C.11)"""
        return (np.sqrt((self.r2 - x) * (x - self.r1)) /
                (2 * np.pi * x * (1-x)) * ((1 + (e - 1) * x) / (1 + (J - 1) * x))**2)

    def compute_overlap_integrate(self, y, J):
        """Computes the overlap in the general case using integrals (unstable when alpha=alpha*)"""
        print('')
        print('Computing Overlap (Integration)')

        e = np.exp(4 * y / self.astar)
        num1 = int(2e3)
        num2 = int(5e3)
        delta = 0.005
        dx1 = delta / num1
        dx2 = (self.r2 - self.r1 - 2*delta) / num2
        X1 = np.linspace(self.r1, self.r1 + delta, num1, endpoint=False) + dx1 / 2
        X2 = np.linspace(self.r1 + delta, self.r2 - delta, num2, endpoint=False) + dx2 / 2
        X3 = np.linspace(self.r2 - delta, self.r2, num1, endpoint=False) + dx1 / 2

        X = [X1, X2, X3[:-2]]
        dx = [dx1, dx2, dx1]
        num = [num1, num2, num1]

        int1 = 0
        int2 = 0

        for k in range(3):
            eta = dx[k]
            for i, x in enumerate(X[k]):
                print('\r[{0}/{1}]'.format(i + 1, num[k]), end="")
                int1 += self.omega1(x, e, J) * eta
                int2 += self.omega2(x, e, J) * eta
            print('')


        A = max(self.a + self.astar - 1, 0) * e / J + int1
        B = max(self.a - self.astar, 0) + max(self.a + self.astar - 1, 0) * e**2 / J**2 + int2
        overlap = A / np.sqrt(self.astar * B)
        print('')

        return overlap


    def compute_overlap_formula(self, y, J):
        """Computes the overlap when alpha = alpha* = 1/2 (direct formula)"""
        print('')
        print('Computing Overlap (Formula)')

        e = np.exp(4 * y / self.astar)
        Jfourth = np.power(J, 1/4)
        Jsqrt = np.sqrt(J)
        num = e * Jfourth
        den = (J**2 + 2 * Jsqrt**3 + 2*e*J + 2*e**2*Jsqrt + e**2) / 2
        return num / np.sqrt(den)


    def optimize(self, end, eta):
        """Computes the solution phi in the high-dimensional limit
        using simple discretization of differential equation
        Computes the overlap if necessary"""
        y = 0
        f = 1
        j = 1

        Y = [0]
        PHI = [0]
        J = [1]

        i = 0

        print('alpha = {0}, alpha* = {1}'.format(self.a, self.astar))
        while y < end:
            if (i+1) % 1000 == 0:
                print('\rStep [{0}/{1}]'.format(i + 1, int(np.ceil(end/eta))), end="")

            phi = self.func(y, j, f)

            j += 4 * eta * np.exp(-2*phi) * (np.exp(4 * y / self.astar) - j) / (self.a * f)
            f += 4 * eta * np.exp(-2*phi) / self.a
            y += eta
            i += 1

            PHI.append(phi)
            Y.append(y)
            J.append(j)

        if self.over:
            overlap = self.compute_overlap(np.array(Y), np.array(J))

            self.overlap = np.array(overlap)

        self.steps = np.array(Y)
        self.Phi = np.array(PHI)
