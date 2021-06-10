# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 13:44:47 2021

@author: phili
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


print("")
print("-----------------------------------------------------------------------------------")
print("-----------------------------------------------------------------------------------")
print("")

Temp  =np.array([20.15, 24.99, 25.00, 37.00, 37.02, 37.26])
Densy =np.array([1.0086, 1.0066, 1.0066, 1.0028, 1.0029, 1.0031]) # g/cm3

plt.figure()
plt.title("Densidad Medio de cultiv vs Temperatura")
plt.scatter(Temp,Densy)
plt.ylim(1.00,1.01)
plt.xlim(19,40)
plt.xlabel("Temp")
plt.ylabel("Density")
plt.grid()
plt.show()

print("")
print("-----------------------------------------------------------------------------------")
print("valor del medio de cultivo a 37° Celsius  es: ", 1.0029,"g/cm3 o", 1002.9 ,"Kg/m3"  )
print("-----------------------------------------------------------------------------------")
print("")
#%%


surfaceTensionMedium = np.array([36.878,37.159,37.709,36.846,38.391,38.526,37.442,38.198,37.486,36.966,37.593,36.651,39.137])
lin1= np.linspace(1,len(surfaceTensionMedium),len(surfaceTensionMedium))



surfaceTensionWater = np.array([69.325,69.297,69.425,69.590,69.325,71.347,69.760 ])
lin2= np.linspace(1,len(surfaceTensionWater),len(surfaceTensionWater))


surfaceTensionWater = np.array([69.325,69.297,69.425,69.590,69.325,71.347,69.760 ])
lin2= np.linspace(1,len(surfaceTensionWater),len(surfaceTensionWater))

surfaceTensionOil = np.array([56.819,53.427,53.844,53.373,52.245,53.038,53.137,53.104])
lin3= np.linspace(1,len(surfaceTensionOil),len(surfaceTensionOil))



#%%

fig = plt.figure()
ax=plt.subplot()

ax.scatter(lin1,surfaceTensionMedium)
ax.plot(lin1,surfaceTensionMedium,label = " Tensión Medio-Aceite")

ax.scatter(lin2,surfaceTensionWater)
ax.plot(lin2,surfaceTensionWater,label = " Tensión Agua-Aire")

ax.scatter(lin3,surfaceTensionOil)
ax.plot(lin3,surfaceTensionOil,label=" Tensión Aceite-Aire")

plt.title("Tensiones superficial")
plt.ylim(20,75)
plt.ylabel("Surface Tension ")
plt.xlabel("Measurements")
plt.xlim(0,17)
ax.legend(loc="best")
plt.grid()
plt.show()





#%%
print("")
print("--------------------------------------------------------------------")
print("Parte 2: comparacion  de densidad es entre OpenDrop y Scian Lab Method")
print("---------------------------------------------------------------------")

#%% TEnsion Agua Aire - OpenDrop / Geometric Method


surfaceTensionWater1 = np.array([70.918,70.514,69.944,70.019,70.157,70.697,70.642,70.554,70.625,70.138,69.849,69.88,69.925,69.256,69.320])
print(len(surfaceTensionWater1))
lin1= np.linspace(1,len(surfaceTensionWater1),len(surfaceTensionWater1))


surfaceTensionWater11 = np.array([ 69.676,68.719,68.797,68.441,69.620,68.617,69.365,68.875,68.656,70.120, 68.56 ,67.369,68.633,67.435,66.402])
#print(len(surfaceTensionWater11))
lin11= np.linspace(1,len(surfaceTensionWater11),len(surfaceTensionWater11))



#%% Tension Agua-Aceite

surfaceTensionOil1 = np.array([53.656,53.311,53.496,53.501,53.884,53.343,52.41,52.945,51.952,51.204,52.274,52.252,51.964,52.236,51.957])
lin2= np.linspace(1,len(surfaceTensionOil1),len(surfaceTensionOil1))
#print(len(surfaceTensionOil1))


surfaceTensionOil11 = np.array([56.062,57.284,57.662,58.951,57.354, 57.014,53.100,54.326,52.220,51.584,52.689, 51.486,54.150,52.947,54.079])
lin22= np.linspace(1,len(surfaceTensionOil11),len(surfaceTensionOil11))
#print(len(surfaceTensionOil1))


#%%
surfaceMedium1 = np.array([33.28, 36.071,35.311, 36.014, 34.769,35.368, 34.625,32.954, 32.934 , 32.842 ,35.879, 34.600, 34.615, 34.553, 36.458])
lin3= np.linspace(1,len(surfaceMedium1),len(surfaceMedium1))


surfaceMedium2 = np.array([34.832,36.946,38.853,35.506,34.939,33.365,35.827,37.563,38.819,36.528,36.519,35.178,38.094,37.863])
lin33= np.linspace(1,len(surfaceMedium2),len(surfaceMedium2))


#%% plots

fig = plt.figure()
ax=plt.subplot()
plt.ylim(9,85)
plt.xlim(0,30)

ax.scatter(lin1,surfaceTensionWater1)
ax.plot(lin1,surfaceTensionWater1,label = "Tensión Aire-Agua//OpenDrop")


ax.scatter(lin11,surfaceTensionWater11)
ax.plot(lin11,surfaceTensionWater11,label = "Tensión Aire-Agua//G.Method")

ax.scatter(lin2,surfaceTensionOil1)
ax.plot(lin2,surfaceTensionOil1,label = " Tensión Aceite-Agua//OpenDrop")

ax.scatter(lin22,surfaceTensionOil11)
ax.plot(lin22,surfaceTensionOil11,label = " Tensión Aceite-Agua//G.Method")


ax.scatter(lin3,surfaceMedium1)
ax.plot(lin3,surfaceMedium1,label = "Tensión Aceite-Medio//OpenDrop")

ax.scatter(lin33,surfaceMedium2)
ax.plot(lin33,surfaceMedium2,label = "Tensión Aceite-Medio//G.Method")


plt.grid()
ax.legend(loc="best")
plt.show()




#%%

print("End of Script")







