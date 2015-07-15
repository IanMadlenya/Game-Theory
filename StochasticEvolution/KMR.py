# coding:UTF-8
"""
Author:Yoshimasa Ogawa
KMR (Kandori-Mailath-Rob) Model
"""
from __future__ import division
import numpy as np
import random as rd
from scipy.stats import binom
import matplotlib.pyplot as plt
import quantecon as qe
import gambit as gb


def kmr_markov_matrix(p, N, epsilon=0, simultaneous=False):
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
    simultaneous : True or False
        sequential or simultaneous

    Returns
    ---------
    P : ndarray(int, ndim=2)
        The transition probability matrix for the KMR dynamics.
    """
    P = np.zeros((N+1, N+1), dtype=float)
    if simultaneous is False:
        P[0][1] = (epsilon) * (1/2)
        P[N][N-1] = (epsilon) * (1/2)
        P[0][0], P[N][N] = 1 - P[0][1], 1 - P[N][N-1]
        for i in range(1, N):
            P[i][i-1] = (i/N) * (epsilon * (1/2) + (1 - epsilon)*(((i - 1)/(N - 1) < p) + ((i - 1)/(N - 1) == p) * (1/2)))
            P[i][i+1] = ((N-i)/N) * (epsilon * (1/2) + (1 - epsilon)*((i/(N - 1) > p) + (i/(N - 1) == p) * (1/2)))
            P[i][i] = 1 - P[i][i-1] - P[i][i+1]
        return P
    else:
        for i in range(0, N+1):
            P[i]= binom.pmf(range(N+1), N, (i/N < p)*epsilon/2+(i/N == p)/2+(i/N > p)*(1-epsilon/2))
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


class KMR_2x2(object):
    """
    Class representing the KMR dynamics with two actions.
    """
    def __init__(self, p, N, epsilon, simultaneous=False):
        self.p = p
        self.N = N
        self.epsilon = epsilon
        self.P = kmr_markov_matrix(p, N, epsilon, simultaneous)
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

    def plot_sample_path(self, ax=None, show=True):
        fig, ax = plt.subplots()
        ax.set_ylim(0, self.N)
        ax.plot(self.sample_path)
        plt.title("Sample path : epsilon = %s, initial = %s" % (self.epsilon, self.sample_path[0]))
        plt.show()

    def plot_stationary_distribution(self):
        self.stationary_dist = self.mc.stationary_distributions[0]
        fig, ax = plt.subplots()
        ax.set_xlim(-0.5, self.N+0.5)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel('State Space')
        ax.set_ylabel('Probability')
        ax.set_title('Stationary Distributions : epsilon = %s' % self.epsilon)
        ax.bar(range(self.N+1), self.stationary_dist, align="center")
        plt.show()

    def plot_empirical_distribution(self, sample_path=None):
        if sample_path is None:
            sample_path = self.sample_path
        fig, ax = plt.subplots()
        n, bins = np.histogram(sample_path, bins=self.N+1, range=(0, self.N))
        ax.set_xlim(-0.5, self.N+0.5)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel('State Space')
        ax.set_ylabel('Probability')
        ax.set_title('Empirical Distributions : epsilon = %s, initial = %s' % (self.epsilon, sample_path[0]))
        ax.bar(range(self.N+1), n/self.ts_length, align="center")
        plt.show()

class KMR_3x3(object):
    """
    Class representing the KMR dynamics with three actions.
    """
    def __init__(self, payoff, N, epsilon):
        self.payoff = payoff
        self.str = len(payoff)
        self.N = N
        self.epsilon = epsilon

    def simulate(self, ts_length, init=None, num_reps=None, allow_return=False):
        u1 = np.random.random(size=ts_length)
        u2 = np.random.random(size=ts_length)
        X = np.empty((ts_length, self.str), dtype=int)
        P = np.empty((3, 3))
        if isinstance(init, list) and len(init) == self.str:
            X[0] = init
        else:
            numlist = [None for i in range(self.str)]
            numlist[0] = rd.randint(0, self.N)
            numlist[1] = rd.randint(0, self.N - numlist[0])
            numlist[2] = self.N - numlist[0] -numlist[1]
            X[0] = numlist
        for t in range(ts_length-1):
            cdf = X[t]/sum(X[t])
            np.cumsum(cdf, out=cdf)
            pl_type_before = np.searchsorted(cdf, u1[t+1])
            P = kmr3x3_cumsum_matrix(P, X[0], self.payoff, self.epsilon)
            pl_type_after = np.searchsorted(P[pl_type_before], u2[t+1])
            X[t+1] = X[t]
            X[t+1][pl_type_before] -= 1
            X[t+1][pl_type_after] += 1
        self.sample_path = X
        if allow_return is True:
            return self.sample_path

    def plot_sample_path(self, ax=None, show=True):
        fig, ax = plt.subplots()
        ax.set_ylim(0, self.N)
        for i in range(3):
            ax.plot(self.sample_path[:, i], label="action%s" % i)
        plt.legend()
        plt.title("Sample path : epsilon = %s, initial = [%s, %s, %s]" % (self.epsilon, self.sample_path[0][0], self.sample_path[0][1], self.sample_path[0][2]))
        plt.show()

def best_response(payoff, numlist):
    """
    Return the best strategy for each player.

    Parameters
    ----------
    payoff : array_like(float, ndim=2)
        The payoff matrix.
    numlist : list
        The list index means each strategy,
        and the key means the number of people
        who have the strategy of index number.

    Returns
    -------
    best_resp : list
        the best strategy.
    """
    numprob = numlist/sum(numlist)
    expe = np.dot(payoff, numprob)
    best_resp = np.where(expe == max(expe))[0]
    return best_resp


def kmr3x3_cumsum_matrix(P, numlist, payoff, epsilon):
    """
    Parameters
    ----------
    P : array_like(float, ndim=2)
        3x3 matrix
    numlist : list
        The list index means each strategy,
        and the key means the number of people
        who have the strategy of index number.
    payoff : array_like(float, ndim=2)

    Returns
    -------
    P : array_like(float, ndim=2)
        3x3 matrix
    """
    best_resp = best_response(payoff, numlist)
    for i in range(3):
        for j in range(3):
            P[i][j] = (j in best_resp)/len(best_resp)*(1-epsilon)+epsilon/3
    np.cumsum(P, axis=-1, out=P)
    return P
