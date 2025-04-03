
#Graphe pour les données du 20 fev avec 40% de glycérol

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Charger le fichier Excel
exp1 = pd.read_excel("exp_40%glycérol_exp1_30°C_dpl_moy.xlsx", engine="openpyxl")
exp2 = pd.read_excel("exp_40%glycérol_exp2_40°C_dpl_moy.xlsx", engine="openpyxl")
exp3 = pd.read_excel("exp_40%glycérol_exp3_50°C_dpl_moy.xlsx", engine="openpyxl")
exp4 = pd.read_excel("exp_40%glycérol_exp4_60°C_dpl_moy.xlsx", engine="openpyxl")
exp5 = pd.read_excel("exp_40%glycérol_exp5_70°C_dpl_moy.xlsx", engine="openpyxl")
exp6 = pd.read_excel("exp_40%glycérol_exp9_100°C_dpl_moy.xlsx", engine="openpyxl")


for i in range(1,7):

    #Création des Xi et Yi
    exec(f'X{i} = np.array([float(i) for i in exp{i}["lagt"]]).reshape(-1,1)')
    exec(f'Y{i} = np.array([float(i) for i in exp{i}["msd"]])')

    # Création du modèle de régression linéaire
    exec(f'model{i} = LinearRegression()')
    
    # Entraînement du modèle
    exec(f'model{i}.fit(X{i}, Y{i})')

    # Coefficients de la régression
    exec(f'a=model{i}.coef_')
    exec(f'b=model{i}.intercept_')
    
    print(f'Coefficients (pente) expérience {i} :', a)
    print(f'Ordonnée à l\'origine (intercept) expérience {i} :', b)

    # Prédictions avec le modèle entraîné
    exec(f'Y{i}_pred = model{i}.predict(X{i})')

# Visualisation du graphe
plt.scatter(X1, Y1, color='blue', label='Mesures à 30°C')
plt.plot(X1, Y1_pred, color='blue', label='Régression à 30°C')

plt.scatter(X2, Y2, color='red', label='Mesures à 40°C')
plt.plot(X2, Y2_pred, color='red', label='Régression à 40°C')

plt.scatter(X3, Y3, color='green', label='Mesures à 50°C')
plt.plot(X3, Y3_pred, color='green', label='Régression à 50°C')

plt.scatter(X4, Y4, color='orange', label='Mesures à 60°C')
plt.plot(X4, Y4_pred, color='orange', label='Régression à 60°C')

plt.scatter(X5, Y5, color='purple', label='Mesures à 70°C')
plt.plot(X5, Y5_pred, color='purple', label='Régression à 70°C')

plt.scatter(X6, Y6, color='brown', label='Mesures à 100°C')
plt.plot(X6, Y6_pred, color='brown', label='Régression à 100°C')

plt.ylim(0, 20)
plt.xlabel('Temps (s)')
plt.ylabel('Déplacement carré moyen (µm²)')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()
