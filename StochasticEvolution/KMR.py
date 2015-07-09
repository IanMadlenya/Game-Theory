# coding:UTF-8
"""
Author:Yoshimasa Ogawa
KMR (Kandori-Mailath-Rob) Model
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe
import gambit as gb


def kmr_markov_matrix(p, N, epsilon=0):
    """
    Generate the transition probability matrix for the KMR dynamics with
    two acitons.

    Parameters
    ----------
    p : scalar(int)
        Probability on mixed strategy nash equilibrium.
    N : scalar(int)
        Number of players.
    epsilon : scalar(int), optional(default=0)
        perturbations.
    Returns
    ---------
    P : ndarray(int, ndim=2)
        The transition probability matrix for the KMR dynamics.
    """
    P = np.zeros((N+1, N+1), dtype=float)
    P[0][1] = (epsilon) * (1/2)
    P[N][N-1] = (epsilon) * (1/2)
    P[0][0], P[N][N] = 1 - P[0][1], 1 - P[N][N-1]
    for i in range(1, N):
        P[i][i-1] = (i/N) * (epsilon * (1/2) + (1 - epsilon)*(((i - 1)/(N - 1) < p) + ((i - 1)/(N - 1) == p) * (1/2)))
        P[i][i+1] = ((N-i)/N) * (epsilon * (1/2) + (1 - epsilon)*((i/(N - 1) > p) + (i/(N - 1) == p) * (1/2)))
        P[i][i] = 1 - P[i][i-1] - P[i][i+1]
    return P


def compute_nash_equilibrium(payoff, same_payoff=False):
    """
    Generate the probabilities on mixed strategy nash equilibrium.

    Parameters
    ----------
    payoff : array_like(float, ndim=2)
        The payoff matrix.
    same_payoff : True or False

    Returns
    -------
    Nash : list
        List of NashProfiles.
    """
    strategy = payoff.shape
    g = gb.Game.new_table([strategy[0], strategy[1]])
    for i in range(strategy[0]):
        for j in range(strategy[1]):
            if same_payoff is False:
                for k in range(strategy[2]):
                    g[i, j][k] = payoff[i][j][k]
            else:
                g[i, j][0] = payoff[i][j]
                g[i, j][1] = payoff[j][i]
    solver = gb.nash.ExternalEnumMixedSolver()
    return solver.solve(g)


class KMR(object):
    """
    Class representing the KMR dynamics with two actions.
    """
    def __init__(self, p, N, epsilon):
        self.p = p
        self.N = N
        self.epsilon = epsilon
        self.P = kmr_markov_matrix(p, N, epsilon)
        self.mc = qe.MarkovChain(self.P)

    def simulate(self, ts_length, init=None, num_reps=None, allow_return=False):
        """
        Simulate the dynamics.
        Parameters
        ----------
        ts_length : scalar(int)
            Length of each simulation.
        init : scalar(int) or array_like(int, ndim=1),
               optional(default=None)
            Initial state(s). If None, the initial state is randomly
            drawn.
        num_reps : scalar(int), optional(default=None)
            Number of simulations. Relevant only when init is a scalar
            or None.
        allow_return : True or False

        Returns
        -------
        X : ndarray(int, ndim=1 or 2)
            Array containing the sample path(s), of shape (ts_length,)
            if init is a scalar (integer) or None and num_reps is None;
            of shape (k, ts_length) otherwise, where k = len(init) if
            init is an array_like, otherwise k = num_reps.
        """
        self.sample_path = self.mc.simulate(ts_length, init, num_reps)
        self.ts_length = ts_length
        if allow_return is True:
            return self.sample_path

    def compute_stationary_distribution(self, allow_return=False):
        # mc.stationary_distributions の戻り値は2次元配列．
        # 各行に定常分布が入っている (一般には複数)．
        # epsilon > 0 のときは唯一，epsilon == 0 のときは複数ありえる．
        # espilon > 0 のみを想定して唯一と決め打ちするか，
        # 0か正かで分岐するかは自分で決める．
        if self.epsilon != 0:
            self.stationary_dist = self.mc.stationary_distributions[0]
            if allow_return is True:
                return self.stationary_dist  # これは唯一と決め打ちの場合
        else:
            print "epsilon is 0"

    def plot_sample_path(self, ax=None, show=True):
        fig, ax = plt.subplots()
        ax.set_ylim(0, self.N)
        ax.plot(self.sample_path)
        plt.show()

    def plot_stationary_distribution(self):
        self.stationary_dist = self.mc.stationary_distributions[0]
        fig, ax = plt.subplots()
        ax.set_xlim(-0.5, self.N+0.5)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel('State Space')
        ax.set_ylabel('Probability')
        ax.set_title('Stationary Distributions')
        ax.bar(range(self.N+1), self.stationary_dist, align="center")
        plt.show()

    def plot_empirical_distribution(self):
        fig, ax = plt.subplots()
        n, bins = np.histogram(self.sample_path, bins=self.N+1, range=(0, self.N))
        ax.set_xlim(-0.5, self.N+0.5)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel('State Space')
        ax.set_ylabel('Probability')
        ax.set_title('Empirical Distributions')
        ax.bar(range(self.N+1), n/self.ts_length, align="center")
        plt.show()
