#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:06:28 2024

@author: gez
"""
#Definiition of avogadro's number
Av = 6.02214076E23

import molmass

#Input for the expected density of the first batch
#FIRST BATCH
dens = 1.5206 #Low Density
#dens = 2.1468  #Mid Density
#dens = 4.9446  #High Density

#SECOND BATCH
#dens = 1.7310 #Low Density
#dens = 2.5276  #Mid Density
#dens = 3.8929  #High Density

#Definition of the dictionary with the number of atoms of each
#FIRST BATCH
ats = [['Be', 3], ['Cl', 56], ['Li', 35], ['Al', 3], ['Mg', 3]] #Low Density
#ats = [['Al', 5], ['Be', 4], ['F', 58], ['K', 2], ['Na', 31]] #Mid Density
#ats = [['K', 2], ['Zn',2], ['Br', 6], ['I', 45], ['Tl', 45]] #High Density

#SECOND BATCH
#ats = [['Al', 6], ['Be', 4], ['Cl', 10], ['F', 48], ['K', 32]] #Low Density
#ats = [['Cl', 55], ['F', 16], ['K', 15], ['Th', 14]] #Mid Density
#ats = [['Cs', 4], ['Gd',20], ['I', 72], ['Sr', 4]] #High Density

#The calculation itself
mass_c = 0
for lis in ats:
    mass_c += (lis[1]*(molmass.Formula(lis[0]).mass))/Av
side = (mass_c/dens)**(1/3)

print('The length of every side is: ', side*(1E8))

#FIRST BATCH
#Low Density 13.804897409734702
#Mid Density 11.687036292662132
#High Density 17.366546593836443

#SECOND BATCH
#Low Density 12.128268623528123
#Mid Density 15.874237388426975
#High Density 20.526363346764224