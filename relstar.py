import numpy as np
import matplotlib.pyplot as plt
import warnings
#warnings.filterwarnings("error")

"""
Solves equations for a (rotating) relativistic star, as based on the method described on page 51 of 'An introduction to the theory of rotating relativistic stars', Gourgoulhon 2010

Going to do axisymmetric, non-rotating first but leave some doors open
to eventually include rotation
"""

from functools import partial

def polytropic_eos(H, gamma=5./3., K=1.e3):
    eps = np.zeros_like(H)
    p = np.zeros_like(H)
    rho = np.zeros_like(H)

    eps[H > 0.] = (np.exp(H[H > 0.]) - 1.0) / gamma
    p[H > 0.] = (gamma * K**(1./gamma) / ((gamma-1.)*(np.exp(H[H > 0.])-1.))) ** (gamma/(gamma+1.))
    rho[H > 0.] = (p[H > 0.]/K)**(1./gamma)

    return eps, p, rho

class Star(object):

    def __init__(self, Hc, Omega, R, eos=None, theta=0.5*np.pi, n=1e4, nIts=100):
        r"""
        Parameters
        ----------
        Hc:     scalar
            central log enthalpy
        Omega:  scalar
            constant angular velocity
        R:      scalar
            guess for radius of final model
        theta:  scalar
            theta at which want to calculate star's profile
        n:      integer
            number of grid points
        nIts:   integer
            number of iterations
        """
        n = int(n)
        self.n = n
        self.Hc = Hc
        self.Omega = Omega
        self.theta = theta
        self.nIts = nIts

        self.R = R

        self.r = np.linspace(0.0, 2.5*R, num=n)
        self.dr = self.r[1] - self.r[0]

        # metric functions
        self.N = np.ones(n)
        self.omega = np.zeros(n)
        self.A = np.ones(n)
        self.B = np.ones(n)

        # thermo variables
        self.U = np.zeros(n)
        self.W = np.zeros(n)
        self.H = Hc * (1. - self.r**2/R**2)

        self.H[self.H < 0.] = 0.
        self.eps = np.zeros(n)
        self.p = np.zeros(n)

        # matter sources
        self.E = np.zeros(n)
        self.p_phi = np.zeros(n)
        self.S_rr = np.zeros(n)
        self.S_thth = np.zeros(n)
        self.S_phiphi = np.zeros(n)
        self.S = np.zeros(n)

        if eos is None:
            self.eos = self.default_eos
        else:
            self.eos = eos

        self.eps, self.p, _ = self.eos(self.H)

    @staticmethod
    def default_eos(H, gamma=5./3., K=1.e3):
        eps = np.zeros_like(H)
        p = np.zeros_like(H)
        rho = np.zeros_like(H)

        eps[H > 0.] = (np.exp(H[H > 0.]) - 1.0) / gamma
        p[H > 0.] = (gamma * K**(1./gamma) / ((gamma-1.)*(np.exp(H[H > 0.])-1.))) ** (gamma/(gamma+1.))
        rho[H > 0.] = (p[H > 0.]/K)**(1./gamma)

        return eps, p, rho


    def evolve_star(self):
        """
        Evolves star's solution through a single iteration
        """
        np.set_printoptions(threshold=np.nan)

        # calculate eos

        self.eps, self.p, _ = self.eos(self.H)

        # compute matter sources

        self.E = self.W**2 * (self.eps + self.p) - self.p
        self.p_phi = self.B * (self.E + self.p) * self.U * self.r * np.sin(self.theta)
        self.S_rr = self.p
        self.S_thth = self.p
        self.S_phiphi = self.p + (self.E + self.p) * self.U**2
        self.S = 3. * self.p + (self.E + self.p) * self.U**2

        # solve the Einstein equations

        nu = np.log(self.N)
        new_nu = np.log(self.N)

        del2omega = np.zeros(self.n)
        del2omega[1:-1] = 0.25 * (self.omega[2:] - self.omega[:-2])**2 / self.dr**2

        delnu_delnuB = np.zeros(self.n)
        try:
            delnu_delnuB[1:-1] = 0.25 * (nu[2:] - nu[:-2]) * (nu[2:] + np.log(self.B[2:]) - nu[:-2] - np.log(self.B[:-2])) / self.dr**2
        except RuntimeWarning:
            print('B: {}'.format(self.B[1:-1]))

        del3_nu = 4. * np.pi * self.A**2 * (self.E + self.S) + 0.5 * self.B**2 * self.r**2 * np.sin(self.theta)**2 * del2omega / self.N**2 - delnu_delnuB

        new_nu[1:-1] = 0.5 * (nu[2:] + nu[:-2] + self.dr * (nu[2:] - nu[:-2])/self.r[1:-1] - self.dr**2 * del3_nu[1:-1])

        delom_delnu3B = np.zeros(self.n)
        delom_delnu3B[1:-1] = 0.25 * (self.omega[2:] - self.omega[:-2]) * (nu[2:] - 3.*np.log(self.B[2:]) - nu[:-2] + 3.*np.log(self.B[:-2])) / self.dr**2

        del3_wrsin = np.zeros(self.n)
        del3_wrsin[1:-1] = -16 * np.pi * self.N[1:-1] * self.A[1:-1]**2 * self.p_phi[1:-1] / (self.B[1:-1]**2 * self.r[1:-1] * np.sin(self.theta)) + self.r[1:-1] * np.sin(self.theta) * delom_delnu3B[1:-1]

        omegarsin = self.omega * self.r * np.sin(self.theta)

        omegarsin[1:-1] = ((omegarsin[2:] + omegarsin[:-2])/self.dr**2 + (omegarsin[2:] - omegarsin[:-2]) / (self.r[1:-1] * self.dr) - del3_wrsin[1:-1]) / (2. / self.dr**2 + 1./(self.r[1:-1] * np.sin(self.theta))**2)

        NBrsin = (self.N * self.B - 1.) * self.r * np.sin(self.theta)

        del2NB = 8. * np.pi * self.N * self.A**2 * self.B * self.r * np.sin(self.theta) * (self.S_rr + self.S_thth)

        NBrsin[1:-1] = 0.5 * (NBrsin[2:] + NBrsin[:-2] + self.dr * 0.5 * (NBrsin[2:] - NBrsin[:-2]) / self.r[1:-1] - self.dr**2 * del2NB[1:-1])

        try:
            lnAnu = np.log(self.A + nu)
        except RuntimeWarning:
            #print('A + nu: {}'.format(self.A + nu))
            lnAnu = np.log(abs(self.A + nu))

        del2nu = np.zeros(self.n)
        del2nu[1:-1] = 0.25 * (nu[2:] - nu[:-2])**2 / self.dr**2

        del2lnAnu = 8. * np.pi * self.A**2 * self.S_phiphi + 0.25 * 3. * self.B**2 * self.r**2 * np.sin(self.theta)**2 * del2omega / self.N**2 - del2nu

        lnAnu[1:-1] = 0.5 * (lnAnu[2:] + lnAnu[:-2] + self.dr * 0.5 * (lnAnu[2:] - lnAnu[:-2]) / self.r[1:-1] - self.dr**2 * del2lnAnu[1:-1])

        # change back to variables we actually want

        self.N[1:-1] = np.exp(new_nu[1:-1])
        self.omega[1:-1] = omegarsin[1:-1] / (self.r[1:-1] * np.sin(self.theta))
        try:
            self.B[1:-1] = (NBrsin[1:-1] / (self.r[1:-1] * np.sin(self.theta)) + 1.) / self.N[1:-1]
        except RuntimeWarning:
            self.N[self.N < 1.e-15] = 1.e-15
            self.B[1:-1] = (NBrsin[1:-1] / (self.r[1:-1] * np.sin(self.theta)) + 1.) / self.N[1:-1]
            #print('N: {}'.format(self.N[1:-1]))

        try:
            self.A[1:-1] = np.exp(lnAnu[1:-1] - new_nu[1:-1])
        except RuntimeWarning:
            #print('lnA - nu: {}'.format(lnAnu[1:-1] - new_nu[1:-1]))
            self.A[1:-1] = np.exp(np.minimum((lnAnu[1:-1] - new_nu[1:-1]), 10.))

        # enforce some outflow boundary conditions for the last data point

        self.N[0] = self.N[1]
        self.omega[0] = self.omega[1]
        self.B[0] = self.B[1]
        self.A[0] = self.A[1]
        new_nu[0] = new_nu[1]

        self.N[-1] = self.N[-2]
        self.omega[-1] = self.omega[-2]
        self.B[-1] = self.B[-2]
        self.A[-1] = self.A[-2]
        new_nu[-1] = new_nu[-2]

        # make sure B is positive
        self.B[self.B < 0.] = abs(self.B[self.B < 0.])


        #print(self.B)

        # find new velocity field

        self.U = (self.Omega - self.omega) * self.r * np.sin(self.theta) * self.B / self.N

        try:
            self.W = 1. / np.sqrt(1. - self.U**2)
        except RuntimeWarning:
            # if U is too big, set it to 0.9
            self.U[self.U > 1.] = 0.9
            self.U[self.U < -1.] = -0.9
            self.W = 1. / np.sqrt(1. - self.U**2)

        self.H = self.Hc + np.log(self.N[0]) - new_nu + np.log(self.W)

        # find the edge of the star and set H to zero there

        self.H[self.H - self.Hc <= 0.] = 0.0
        self.H[0] = self.Hc
        self.H[self.H > 10.] = 10.

    def make_star(self):
        """
        evolves star through self.nIts iterations
        """

        for i in range(int(self.nIts)):
            #print('Step {}'.format(i))
            self.evolve_star()

    def evolve_tov(self, K=1., gamma=5./3.):
        """
        Finds tov solution
        """

        # calculate eos

        nu = np.log(self.N)
        new_nu = np.log(self.N)

        m = np.zeros(self.n)
        m[1:-1] = 2. * np.pi * (self.r[2:]**2 * self.eps[2:] - self.r[:-2]**2 * self.eps[:-2]) / self.dr
        m[-1] = m[-2]

        dnu_dr = (m / self.r**2 + 4. * np.pi * self.r * self.p) / (1. - 2. * m / self.r)

        new_nu[1:-1] = 0.5 * (dnu_dr[2:] - dnu_dr[:-2]) / self.dr

        self.p[1:-1] = -0.5 * ((self.eps[2:] + self.p[2:]) * dnu_dr[2:] - (self.eps[:-2] + self.p[:-2]) * dnu_dr[:-2]) / self.dr
        self.p[-1] = self.p[-2]

        self.N[1:-1] = np.exp(new_nu[1:-1])
        self.N[-1] = self.N[-2]

        self.eps = p**((gamma-1.)/gamma) / ((gamma-1.) * K**(-1./gamma))

    def make_tov(self, K=1., gamma=5./3.):
        """
        Evolve tov solution
        """
        for i in range(int(self.nIts)):
            #print('Step {}'.format(i))
            self.evolve_star()

        self.H[:] = 0.
        self.H[self.p > 0.] = np.exp(1. + gamma / (self.p[self.p > 0.]**(1.-1./gamma) * K**(-1./gamma) * (gamma-1.)))

    def plot_star(self):
        """
        Plots H, p and N against radius
        """
        self.eps, self.p, rho = self.eos(self.H)

        # calculate mass
        # find cell-centred densities
        av_rho = np.zeros_like(rho)
        av_rho[:-1] = 0.5 * (rho[:-1] + rho[1:])
        M = np.zeros_like(rho)
        for i in range(int(self.n)-1):
            M[i] = 4. * np.pi * np.sum((self.r[1:i+2]**3 - self.r[:i+1]**3) * av_rho[:i+1]) / 3.
        M[-1] = M[-2]

        # normalise everything
        H = self.H / max(self.H[1:])
        p = self.p / max(self.p[1:])
        N = self.N / max(self.N[1:])
        M = M / max(M[1:])
        B = self.B / max(self.B[1:])
        A = self.A / max(self.A[1:])

        plt.clf()
        plt.rc("font", size=15)
        ax = plt.subplot(111)
        lines = ax.plot(self.r, H, '--', self.r, B, ':', self.r, A, '--', self.r, M, '-.')
        plt.setp(lines, linewidth=2)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        lgd = plt.legend([r'$H$', r'$B$', r'$A$', r'$M$'], loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel(r'$r$')
        plt.ylim([0., 1.1])
        plt.xlim([0., self.r[-1]])
        plt.show()



if __name__ == "__main__":

    Hc = 0.2262
    Omega = 0.0
    R = 7.e-4
    gamma = 8./3.0
    K = 1.e-5

    eos_star = partial(polytropic_eos, gamma=gamma, K=K)

    star = Star(Hc, Omega, R, eos=eos_star, nIts=1e3)
    star.make_star()
    star.plot_star()

    tov_star = Star(Hc, Omega, R, eos=eos_star, nIts=1e3)
    tov_star.make_tov(K=K, gamma=gamma)
    tov_star.plot_star()
