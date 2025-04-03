# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 12:29:01 2025

@author: julia
"""

from statistics import mean
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random as rd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy.linalg as la



tableau = pd.read_excel("C:/Users/julia/Desktop/ecole/Ltrois_PF/projet_tut/lol.xlsx")
Lx = tableau['x'].to_numpy() #conversion de la colonne des x en tableau numpy
Ly = tableau['y'].to_numpy()
L_ind = tableau['particle'].to_numpy() #liste des indices

n1 = len(Lx) # nombre de frames au total qu'on a collecté

L = []
for i in range(n1):
    L.append([L_ind[i],Lx[i],Ly[i]])
    

L_pos = L #contient toutes les positions des particules i
#print(L_position)

L1 = [[]]
j = 0
n_temp = L_pos[0][0]
for i in range(n1):
    if L_pos[i][0] == n_temp:
        L1[j].append([L_pos[i][1],L_pos[i][2]])
    else :
        j+=1
        n_temp = L_pos[i][0]
        L1.append([])

  

def calcul_distances(L):
    # Initialisation de la liste de résultats
    L2 = []
    
    # Parcourir chaque sous-liste dans L
    for sous_liste in L:
        distances = []
        # Parcourir les éléments de chaque sous-liste (sauf le dernier)
        for j in range(len(sous_liste) - 1):
            # Calcul de la distance entre deux points consécutifs
            x1, y1 = sous_liste[j]
            x2, y2 = sous_liste[j + 1]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            distances.append(distance)
        L2.append(distances)
    
    return L2

M = [[[0,1],[8,6]],[[3,9],[4,5],[6,0]],[[5,5],[8,1]]]

L2 = calcul_distances(L1)

def moyenne_elements(L):
    result = [sum(x[i] for x in L) / len(L) for i in range(len(L[0]))]
    return result


#Tentative d'enlever le drift à la main mais ça n'a rien donné

L_nb_frames = []
for i in range(len(L1)):
    L_nb_frames.append(len(L1[i]))

def f(L,k):
    L1 = []
    for i in range(len(L)):
        if k <= L[i]:
            L1.append(L[i])
    return L1

pos_intermediaire = []
for i in range(len(L1)):
    if 203 <= len(L1[i]):
        pos_intermediaire.append(L1[i])

#pos_intermediaire retourne la position de toutes les particules qui ont été suivies au moins pendant 203 frames

def get_k_premiers_elements(L, k):
    result = [lst[:k] for lst in L]
    return result

new_pos = get_k_premiers_elements(pos_intermediaire,203)

n = len(new_pos)
n2 = len(new_pos[0])

for i in range(n):
    for j in range(1,len(new_pos[i])):
        new_pos[i][j][0] -= new_pos[i][0][0]
        new_pos[i][j][1] -= new_pos[i][0][1]
        
for i in range(n):
    new_pos[i][0][0] = 0
    new_pos[i][0][1] = 0
    
A = [] #contient tous les déplacements carrés des particules
for k in range(n):
    A.append([])
for i in range(0,len(new_pos)):
    for j in range(len(new_pos[i])):
        A[i].append(new_pos[i][j][0]**2 + new_pos[i][j][1]**2)
        
B = [] #contient les déplacements de toutes les particules
for k in range(n):
    B.append([])
for i in range(0,len(new_pos)):
    for j in range(len(new_pos[i])):
        B[i].append(np.sqrt(new_pos[i][j][0]**2 + new_pos[i][j][1]**2))
        
deplacement_carre_moy = moyenne_elements(A)
C = moyenne_elements(B)

def calcul_diff(L1, L2):
    # Vérification que les deux listes ont la même longueur
    if len(L1) != len(L2):
        raise ValueError("Les deux listes doivent avoir la même longueur.")
    
    # Calcul de la nouvelle liste L
    L = [L1[i] - L2[i]**2 for i in range(len(L1))]
    
    return L



deplacement_carre_moy_reel = calcul_diff(deplacement_carre_moy,C)

y1 = deplacement_carre_moy
#y2 = [np.sqrt(y) for y in y1]
t = np.arange(1,204)*(30/444)

plt.plot(t,y1)
plt.show()


def x1(t):
    return t
def x2(t):
    return t**2

D = np.ones((203,3))

D[:,1] = x1(t)
D[:,2] = x2(t)

beta = la.solve(D.T @ D, D.T @ y1)
