import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import os
import glob
import pandas as pd
import random as rd
from matplotlib.colors import LogNorm

mpl.use('Agg')

################################################
######## Initialisation des variables ##########
################################################

"""/!\ dt < dx, dy"""
dx = dy = 0.05              #pas espace
Lx = 7                      #1/2 longueur boite x (entier)
Ly = 10                     #1/2 longueur boite y (entier)

#Lx = 7                      #lentille CV
#Ly = 13                     #lentille CV

x = np.arange(-Lx, Lx, dx)
lx = len(x)                 #nb indice x
y = np.arange(-Ly, Ly, dy)
ly = len(y)                 #nb indice y

X, Y = np.meshgrid(x, y)

dt = 0.005                  #pas temps
tf = 13                     #temps final
t = np.arange(0, tf, dt)
pas_img = 4                 #pas de images enregistées
digit = 4

x0 = int(ly/16)             #position source (initiale)
y0 = int(lx/2)
K = 60
omega = 15                  #pulsation source
acc = 5                     #acceleration source (def mvt_acc)

print(lx, ly)
################################################
############### Ondes initiales ################
################################################

psi0 = np.zeros((lx, ly))
psi1 = np.copy(psi0)

psi_arr = [psi0, psi1]

################################################
################## Fonctions ###################
################################################

############ nom image ###############

def name(i,digit):
    """Fonction nommant les images dans le fichier img"""

    i = str(i)
    while len(i)<digit:
        i = '0'+i
    i = 'img/'+i+'.png'

    return(i)

############ initialisation c ###############

def rd_block():
    """"
    Valeur célérité aléatoire par bloc.
    Attention, nx doit être divisible par lx et ny par ly"""

    N_sq_x = int(Lx)
    N_sq_y = int(Ly)

    nx = int(lx/N_sq_x)     #nb de points d'un bloc suivant x
    ny = int(ly/N_sq_y)     #nb de points d'un bloc suivant y

    list_block = []
    for i in range(N_sq_x):
        list_block_i = []
        for j in range(N_sq_y):
            list_block_i.append(np.ones((ny, nx)) * rd.uniform(0.5, 1.5))
        list_block.append(list_block_i)

    A = np.block(list_block)
    A[y0 - int(ny/2): y0 + int(ny/2), x0 - int(nx/2) : x0 + int(nx/2)] = np.ones((ny, nx))  #célérité neutre au niveau de la source

    return A[1:lx-1, 1:ly-1]

def fente():
    """Expérience Fente"""
    largeur = 5
    x_f = int(ly/2)         #position en x de la fente

    A = np.ones((lx-2, ly-2)) * 1.5
    A[0: int(lx/2) - largeur, x_f-5:x_f+5] = np.zeros((int(lx/2) - largeur, 10))
    A[int(lx/2) + largeur - 2:, x_f-5:x_f+5] = np.zeros((int(lx/2) - largeur, 10))

    return A

def double_fente():
    """Expérience Fente. Prendre Ly = 13"""
    largeur = 12
    x_f = int(ly/2)         #position en x de la fente
    y_m = int(15)           #largeur entre les deux fentes

    A = np.ones((lx-2, ly-2)) * 2
    A[0: int(lx/2) - largeur - y_m, x_f-2:x_f+2] = np.zeros((int(lx/2) - largeur - y_m, 4))
    A[int(lx/2) - y_m: int(lx/2) - 2 + y_m, x_f-2:x_f+2] = np.zeros((2 * y_m - 2, 4))
    A[int(lx/2) + largeur - 2 + y_m:, x_f-2:x_f+2] = np.zeros((int(lx/2) - largeur - y_m, 4))

    return A

def lentille_CV():
    """Expérience de la lentille CV"""

    c = np.ones((lx-2, ly-2))
    R1 = 13/dx                      #rayon de courbure 1
    R2 = 13/dx                      #rayon de courbure 2
    x0_cv = int(-1 * ly/8)          #centre cercle 1
    x0_cv2 = int(5.5*ly/8)          #centre cercle 2

    for k in range(0, ly-2):
        for l in range(0, lx-2):
            if (k-x0_cv)**2 + (l-int(lx/2))**2 <= R1**2  and  (k-x0_cv2)**2 + (l-int(lx/2))**2 <= R2**2 :
                #print(k, l)
                c[l, k] = 0.5

    return c

############ fonction source ###############

def mvt_acc(psi):
    """Effet doppler"""

    psi[int(lx/2)-2:int(lx/2)+2, int(acc/2 * t[i]**2 + x0)-2 : int(acc/2 * t[i]**2 + x0)+2] += np.ones((4, 4)) * np.sin(omega * t[i])
    return psi

def random(psi):
    """Pluie"""

    if i%K*pas_img == 0:
        x, y = rd.randint(2, lx-2), rd.randint(2, ly-2)
        psi[x-2:x+2, y-2:y+2] += np.ones((4, 4)) * np.sin(np.pi * i / K + np.pi/2)
    return (psi)

def plane(psi):
    """Onde plane"""
    psi[2:lx-2, x0] += np.ones(lx-4) * np.sin(omega * t[i])
    return psi

def circ(psi):
    """Source ponctuelle"""
    psi[y0-2:y0+2, x0-2:x0+2] += np.ones((4, 4)) * 2 *  np.sin(omega * t[i])
    return psi

def impulsion(psi):
    """Impulsion unique"""
    if i == 20:
        psi[y0-2:y0+2, x0-2:x0+2] += np.ones((4, 4))
    return psi

############ fonction aux bords ###############

def tor_x(psi):
    """Conditions périodiques en x"""

    psi[:, 0] += psi[:, -2]
    psi[:, -1] += psi[:, 2]
    return psi

def tor_y(psi):
    """Conditions périodiques en y"""

    psi[0, :] += psi[-2, :]
    psi[-1, :] += psi[2, :]
    return psi

def attenuation_x(psi, psi1):

    psi[1:lx-1, 0] += psi1[1:lx-1, 1] + const1 * (psi[1:lx-1, 1] - psi1[1:lx-1, 0])
    psi[1:lx-1, -1] += psi1[1:lx-1, -2] + const2 * (psi[1:lx-1, -2] - psi1[1:lx-1, -1])
    return psi

def attenuation_y(psi, psi1):

    psi[0, 1:ly-1] += psi1[1, 1:ly-1] + const3 * (psi[1, 1:ly-1] - psi1[0, 1:ly-1])
    psi[-1, 1:ly-1] += psi1[-2, 1:ly-1] + const4 * (psi[-2, 1:ly-1] - psi1[-1, 1:ly-1])
    return psi

############ fonction update ###########

def update(psi0, psi1):

    psi = np.zeros((lx, ly))

    psi[1:lx-1, 1:ly-1] = (c * dt/dx)**2 * (psi1[0:lx-2, 1:ly-1] + psi1[2:lx, 1:ly-1]) + \
        (c * dt/dy)**2 * (psi1[1:lx-1, 0:ly-2] + psi1[1:lx-1, 2:ly]) + \
        2 * (1 - (c * dt/dx)**2 - (c * dt/dy)**2) * psi1[1:lx-1, 1:ly-1] - psi0[1:lx-1, 1:ly-1]

    #psi = tor_x(psi)
    #psi = tor_y(psi)
    psi = attenuation_x(psi, psi1)
    psi = attenuation_y(psi, psi1)

    #psi = mvt_acc(psi)
    #psi = random(psi)
    #psi = plane(psi)
    psi = circ(psi)
    #psi = impulsion(psi)

    psi0 = np.copy(psi1)
    psi1 = np.copy(psi)


    return psi, psi0, psi1

################################################
######### initialisation célérité ##############
################################################

#c = rd_block()
#c = fente()
c = double_fente()
#c = np.ones((lx-2, ly-2)) * 1.5
#c = lentille_CV() * 3

kappa1 = c[:, 1] * dt/dx
const1 = (kappa1 - 1)/(kappa1 +1)
kappa2 = c[:, -1] * dt/dx
const2 = (kappa2 - 1)/(kappa2 +1)
kappa3 = c[1, :] * dt/dy
const3 = (kappa3 - 1)/(kappa3 +1)
kappa4 = c[-1, :] * dt/dy
const4 = (kappa4 - 1)/(kappa4 +1)

################################################
################## Animation ###################
################################################

psi_arr = [psi0, psi1]
for i in range(1, len(t)-1):

    psi, psi0, psi1 = update(psi0, psi1)
    if i%pas_img==0:
        psi_arr.append(psi)


A_max = np.max(psi_arr)
A_min = np.min(psi_arr)

extension="img/*.png"
for f in glob.glob(extension):
  os.remove(f)

for i in range(0, len(psi_arr)-1):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.axis('equal')

    im = ax.imshow(psi_arr[i], extent = (-Ly, Ly, -Lx, Lx), vmin = A_min/2, vmax = A_max/2, cmap = "bwr", interpolation = "gaussian")
    im1 = ax.imshow(c, extent = (-Ly, Ly, -Lx, Lx), cmap = "Greys_r", alpha = 0.1)
    #im = ax.imshow(psi_arr[i], extent = (-Ly, Ly, -Lx, Lx), cmap = "bwr", interpolation = "gaussian")

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axis('off')
    #ax.set_title("v = " + str(acc * t[i * pas_img])[0:4])

    cbar = fig.colorbar(im, ax=ax, fraction=0.0323, pad=0.04)
    #cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_ticks([])
    cbar.ax.tick_params(size=0)

    name_pic = name(int(i), digit)
    plt.savefig(name_pic, bbox_inches='tight', dpi=300)

    ax.clear()
    plt.close(fig)

    print(i/len(psi_arr))

print("Images successed")

  #ffmpeg -r 10 -i img/%04d.png -vcodec libx264 -y -an test.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"
    #ffmpeg -r 50 -i img/%04d.png -vcodec libx264 -y -an test.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"

