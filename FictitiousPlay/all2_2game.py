#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from random import uniform
fig, ax = plt.subplots()

NUM = 1

#　自分で入力したい時
if NUM == 0:
    print '2×2の利得表をひとつずつ記入してください。'
    pr_0_00 = float(raw_input('プレイヤー0左上:'))
    pr_0_01 = float(raw_input('プレイヤー0右上:'))
    pr_0_10 = float(raw_input('プレイヤー0左下:'))
    pr_0_11 = float(raw_input('プレイヤー0右下:'))
    pr_1_00 = float(raw_input('プレイヤー1左上:'))
    pr_1_01 = float(raw_input('プレイヤー1右上:'))
    pr_1_10 = float(raw_input('プレイヤー1左下:'))
    pr_1_11 = float(raw_input('プレイヤー1右下:'))
    titlename = raw_input('タイトル名:')
    #player
    p0 = np.array([[pr_0_00,pr_0_01],[pr_0_10,pr_0_11]])
    p1 = np.array([[pr_1_00,pr_1_10],[pr_1_01,pr_1_11]])

#Maching Pennies
if NUM == 1:
    titlename = 'Matching Pennies'
    p0 = np.array([[1,-1],[-1,1]])
    p1 = np.array([[-1,1],[1,-1]])

#Coordination Game
if NUM == 2:
    titlename = 'Coordination Game'
    p0 = np.array([[4,0],[3,2]])
    p1 = np.array([[4,0],[3,2]])

#Prisoner's Dilemma
if NUM == 3:
    titlename = 'Prisoners Dilemma'
    p0 = np.array([[-5,0],[-10,-3]])
    p1 = np.array([[-5,0],[-10,-3]])
    
na = []

class Fictplay:
    
    def act(self, x, y):
        if x>y:
            return 0
        elif x<y:
            return 1
        else:
            return random.randint(0,1)
        
    def belief(self, n = 1000):
        b0 = [uniform(0, 1)]
        b1 = [uniform(0, 1)]
        for i in range(n-1):
            # belief
            bx0 = np.array([1-b0[i], b0[i]])
            bx1 = np.array([1-b1[i], b1[i]])
            # expected payoff 
            ep0 = np.dot(p0, bx0)
            ep1 = np.dot(p1, bx1)
            # action
            a0 = self.act(ep0[0], ep0[1])
            a1 = self.act(ep1[0], ep1[1])
            # append
            b0.append(b0[i]+(a1-b0[i])/(i+2))
            b1.append(b1[i]+(a0-b1[i])/(i+2))
        self.belief0 = b0
        self.belief1 = b1
            
    def move(self):
        plt.plot(self.belief0, 'r-', label = "player_0" )
        plt.plot(self.belief1, 'b-', label = "player_1")
        plt.legend()
        plt.title(titlename)
        plt.xlabel("time")
        plt.ylabel("belief")
        plt.show()
        
    def nash(self, ts = 100, n = 1000):
        for i in range(ts):
            self.belief()
            na.append(self.belief0[n-1])
    
    def hist(self):
        plt.xlim([0,1])
        plt.hist(na)
        plt.legend()
        plt.show()