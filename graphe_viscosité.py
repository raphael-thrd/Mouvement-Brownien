# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 10:11:33 2025

@author: Joanny
"""

#Graphe viscosité en fct de la température

import numpy as np
import matplotlib.pyplot as plt

#résultats protocole 1

X1=np.array([20,40,50,70])
Y1=np.array([0.076,0.0033,0.051,0.24]) #valeurs expérimentales
Y1_inc=np.array([0.045,0.0004,0.017,0.35]) #incertitudes
Y1_tab=np.array([1.76,1.07,0.879,0.635]) #valeur tabulées

#résultats protocole 2
X2=np.array([30,40,50,60,70,100])
Y2=np.array([0.57,0.12,0.36,0.0099,0.46,0.22])
Y2_inc=np.array([2.99,0.09,0.84,0.0007,1.27,0.26])
Y2_tab=np.array([2.72,2.07,1.62,1.3,1.09,0.668])


plt.errorbar(X1, Y1, yerr=Y1_inc, fmt='bo', capsize=5, ecolor='blue', elinewidth=1, label='Viscosité expérimentale (protocole 1)')
plt.plot(X1, Y1_tab, 'bx', label='Viscosité théorique (protocole 1)')

plt.errorbar(X2, Y2, yerr=Y2_inc, fmt='ro', capsize=5, ecolor='red', elinewidth=1, label='Viscosité expérimentale (protocole 2)')
plt.plot(X2, Y2_tab, 'rx', label='Viscosité théorique (protocole 2)')

plt.xlabel('Température en °C')
plt.ylabel('Viscosité en mPa.s')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()