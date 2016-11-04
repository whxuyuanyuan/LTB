# Load usefull libraries:
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


# -|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#
# Define global variables:

global MatPent, MatPentEdge, Map2a, Map2b, r0


# -|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#
# Load data:

r0 = np.load('tmp/r0.npy');
r1 = np.load('tmp/r1.npy');
r2 = np.load('tmp/r2.npy')
MatPent = np.load('tmp/MatPent.npy');
MatPentEdge = np.load('tmp/MatPentEdge.npy');
MatPentS = np.load('tmp/MatPentS.npy');
MatPentL = np.load('tmp/MatPentL.npy')


fileInput = open('tracking.txt', 'r')
px = []
py = []
angle = []
step = 0;
for line in fileInput:
    x = []
    y = []
    a = []
    numbers = line.split()
    for i in range(len(numbers)):
        if i % 4 == 1:
            x.append(float(numbers[i]))
        if i % 4 == 2:
            y.append(float(numbers[i]))
        if i % 4 == 3:
            a.append(float(numbers[i]))
    px.append(x)
    py.append(y)
    angle.append(a)

os.system('mkdir tracking')
os.system('mkdir tracking/pictures')

for i in range(3):
    fig, ax = plt.subplots()
    for itPt in range(0, len(px)):
        I1 = px[itPt][i]
        I2 = py[itPt][i]
        A0 = -angle[itPt][i] / 180 * m.pi;
        ax.text(I2, I1, str(itPt + 1), size=5, ha='center', va='center')
        plt.plot([I2 + r0 * m.cos(A0 + 0 * 2 * m.pi / 5), I2 + r0 * m.cos(A0 + 1 * 2 * m.pi / 5),
                  I2 + r0 * m.cos(A0 + 2 * 2 * m.pi / 5), I2 + r0 * m.cos(A0 + 3 * 2 * m.pi / 5),
                  I2 + r0 * m.cos(A0 + 4 * 2 * m.pi / 5), I2 + r0 * m.cos(A0 + 5 * 2 * m.pi / 5)],
                 [I1 + r0 * m.sin(A0 + 0 * 2 * m.pi / 5), I1 + r0 * m.sin(A0 + 1 * 2 * m.pi / 5),
                  I1 + r0 * m.sin(A0 + 2 * 2 * m.pi / 5), I1 + r0 * m.sin(A0 + 3 * 2 * m.pi / 5),
                  I1 + r0 * m.sin(A0 + 4 * 2 * m.pi / 5), I1 + r0 * m.sin(A0 + 5 * 2 * m.pi / 5)], linewidth=0.8,
                  color='b')
    plt.xlim([0, 4035])
    plt.ylim([0, 3027])
    plt.axis('equal')
    plt.axis('off')
    plt.title('image_%04d'%(i+1))
    plt.savefig('tracking/pictures/image_%04d'%(i) + '.png', dpi=250)   
    plt.close()
    
os.system('ffmpeg -r 10 -f image2 -i tracking/pictures/image_%04d.png -qscale 1 -vf scale=1500:1000 tracking/tracking.avi >/dev/null 2>&1')


