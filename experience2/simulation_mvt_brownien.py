# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 12:25:04 2025

@author: julia
"""
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy.linalg as la

'''
def diffusion(N,lmp,tau):
    for i in range(N):
        tor = Turtle()
        tor.pendown()
        for i in range(tau):
            distance,theta = rd.uniform(0,lmp), rd.uniform(0,360)
            if 0 < theta < 180:
                tor.speed(0)
                tor.left(theta)
                tor.forward(distance)
            if 180 <= theta < 360:
                tor.speed(0)
                tor.right(np.pi/2)
                tor.forward(distance)
    tor.penup()
'''


def position(l,tau): #renvoie une liste des positions successives d'une particule
    L = [[0,0],]
    for k in range(tau):
        L.append([0,0])
    for i in range(1,tau+1):
        d,angle = rd.uniform(0,l), rd.uniform(0,360)
        L[i][0] = L[i-1][0] + d*np.cos(angle)
        L[i][1] = L[i-1][1] + d*np.sin(angle)
    return L

def position_Nparticules(N,l,tau): #renvoie une liste de liste qui contient les positions successives des N particules
    positionN = []
    for i in range(N):
        positionN.append(position(l,tau))
    return positionN


def deplacement_carre(position): #renvoie les déplacements carrés d'une seule particule aux instants tn
    L = []
    for i in range(len(position)):
        L.append(position[i][0]**2+position[i][1]**2)
    return L


def deplacement_carre_Nparticules(N,l,tau): #renvoie une liste de liste avec tous les déplacements carrés des N particules
    L = position_Nparticules(N,l,tau)
    n = len(L)
    L1 = []
    for i in range(n):
        L1_i = L[i]
        L2_i = deplacement_carre(L1_i)
        L1.append(L2_i)
    return L1
     
def moyenne_elements(L):
    result = [sum(x[i] for x in L) / len(L) for i in range(len(L[0]))]
    return result
'''
T1 = deplacement_carre_Nparticules(500,2.28943e-6,444)

y = moyenne_elements(T1)
t = np.arange(1,446)*(30/444)

plt.plot(t,y,'ob')
plt.xlabel('temps en s')
plt.ylabel('Déplacement carré moyen en m²')
plt.show()
'''
