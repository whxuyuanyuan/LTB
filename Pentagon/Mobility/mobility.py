#Load usefull libraries:
import numpy as np
import math as m
from scipy import misc
from scipy import optimize
from scipy import stats
from scipy import ndimage as nd
import matplotlib.pyplot as plt
import matplotlib.patches as pth
from pylab import ginput
import cv2
import os
from utilities import *

# load the radius of pentagon
#r0 = np.load('tmp/r0.npy')
r0 = 62

N = 835  # number of particles
N_80 = int(N * 0.8)
N_60 = int(N * 0.6)
N_40 = int(N * 0.4)

# particle index starts from 0 and cycle number starts from 0
x = []
y = []
a = []

for i in range(1, N + 1):
    x0, y0, a0 = readFile(i)
    x.append(x0)
    y.append(y0)
    a.append(a0)

for i in range(1, 10):
    interval = []
    temp = []
    for j in range(0, N):
        dis = distance(x[j][i], y[j][i], x[j][i - 1], y[j][i - 1])
        interval.append(dis)
        temp.append(dis)
    temp.sort()
    indices1, = np.where(interval > temp[N_80 - 1])
    indices2, = np.where((interval <= temp[N_80 - 1]) & (interval > temp[N_60 - 1]))
    indices3, = np.where((interval <= temp[N_60 - 1]) & (interval > temp[N_40 - 1]))
    indices4, = np.where(interval <= temp[N_40 - 1])

    # red
    for p in range(len(indices1)):
        index = indices1[p]
        I1 = x[index][i]
        I2 = y[index][i]
        A0 = -a[index][i] / 180 * m.pi
        poltPentagon(I1, I2, A0, r0, 'red')
    # green
    for p in range(len(indices2)):
        index = indices2[p]
        I1 = x[index][i]
        I2 = y[index][i]
        A0 = -a[index][i] / 180 * m.pi
        poltPentagon(I1, I2, A0, r0, 'green')
    # green
    for p in range(len(indices3)):
        index = indices3[p]
        I1 = x[index][i]
        I2 = y[index][i]
        A0 = -a[index][i] / 180 * m.pi
        poltPentagon(I1, I2, A0, r0, 'blue')
    # black
    for p in range(len(indices4)):
        index = indices4[p]
        I1 = x[index][i]
        I2 = y[index][i]
        A0 = -a[index][i] / 180 * m.pi
        poltPentagon(I1, I2, A0, r0, 'black')
    print(len(indices1))
    print(len(indices2))
    print(len(indices3))
    print(len(indices4))
    print('**************')
    plt.xlim([0, 4000])
    plt.ylim([0, 3000])
    plt.axis('equal')
    plt.axis('off')
    plt.title('cycle '+'%04d' % i)
    plt.savefig('cycle_%04d.png' % i, dpi=250)
    plt.close()


