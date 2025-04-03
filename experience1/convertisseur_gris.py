#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 19:29:03 2025

@author: raph


Convertisseur d'image couleur en niveaux de gris'
"""

from PIL import Image

for j in [2,3,4,5,8,9]:

        for i in range(1,444):
                if i<=9:
                        img = Image.open(f"C:/Users/Joanny/OneDrive - umontpellier.fr/Scolaire/Licence 3 PF/projet tutoré/projet tutoré 20fev/exp{j}_20fev/img/image_000{i}.tiff")

                if i>=10 and i<=99:
                        img = Image.open(f"C:/Users/Joanny/OneDrive - umontpellier.fr/Scolaire/Licence 3 PF/projet tutoré/projet tutoré 20fev/exp{j}_20fev/img/image_00{i}.tiff")

                if i>=100 :
                        img = Image.open(f"C:/Users/Joanny/OneDrive - umontpellier.fr/Scolaire/Licence 3 PF/projet tutoré/projet tutoré 20fev/exp{j}_20fev/img/image_0{i}.tiff")


                imgGray = img.convert("L")
                imgGray.save(f"C:/Users/Joanny/OneDrive - umontpellier.fr/Scolaire/Licence 3 PF/projet tutoré/projet tutoré 20fev/exp{j}_20fev/img_gris/image{i}.tiff")


