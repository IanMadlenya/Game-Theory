# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mc_tools import mc_compute_stationary, mc_sample_path
from discrete_rv import DiscreteRV
import random
import fractions

payoff = np.array([[[4, 4], [0, 3]],
                   [[3, 0], [2, 2]]])
p_0 = np.array([[payoff[0][0][0], payoff[0][1][0]],
                [payoff[1][0][0], payoff[1][1][0]]])
n = 10
t = 10000
epsilon = 0.2
r = random.uniform(0, 1)
psi = (r, 1-r)

P = np.zeros([n+1, n+1]) 
for i in range(1, n+1): #行動１をとっている人が選ばれて行動０に変更する
    num0 = fractions.Fraction(i-1, n-1)
    num1 = fractions.Fraction(i, n)
    ratio = np.array([1.0-num0, num0])
    exp = np.dot(p_0, ratio)
    if exp[0] > exp[1]:
        P[i][i-1] = num1*((1.0-epsilon) + epsilon*0.5)
    else:
        P[i][i-1] = num1*(epsilon*0.5)
for i in range(n): #行動0をとっている人が選ばれて行動1に変更する
    num0 = fractions.Fraction(i, n-1)
    num1 = fractions.Fraction(i, n)
    ratio = np.array([1-num0, num0])
    exp = np.dot(p_0, ratio)
    if exp[0] < exp[1]:
        P[i][i+1] = (1-num1)*((1-epsilon) + epsilon*0.5)
    else:
        P[i][i+1] = (1-num1)*(epsilon*0.5)
for i in range(1, n): #行動を変えない
    P[i][i] = 1 - P[i][i-1] - P[i][i+1]
P[0][0] = 1 - P[0][1]
P[n][n] = 1 - P[n][n-1]

X = mc_sample_path(P, psi, t)
plt.plot(X)
plt.show()