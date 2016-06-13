# Load usefull libraries:
import numpy as np
import math as m
from scipy import misc
from scipy import optimize
from scipy import stats
from scipy import ndimage as nd
import matplotlib.pyplot as plt
from pylab import ginput
import cv2
import os
from cv2 import imshow


# Load usefull libraries:
import numpy as np
import math as m
from scipy import misc
from scipy import optimize
from scipy import stats
from scipy import ndimage as nd
import matplotlib.pyplot as plt
from pylab import ginput
import colorsys
import os
import matplotlib


os.system('mkdir images')

# an array of parameters, each of our curves depend on a specific
# value of parameters
parameters = np.linspace(0, 6, 7)

# norm is a class which, when called, can normalize data into the
# [0.0, 1.0] interval.
norm = matplotlib.colors.Normalize(
    vmin=np.min(parameters),
    vmax=np.max(parameters))

# choose a colormap
c_m = matplotlib.cm.rainbow

# create a ScalarMappable and initialize a data structure
s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
s_m.set_array([])



r0 = np.load('../tmp/r0.npy')
r0 *= 0.98

contact_numbers = []
for i in range(0, 7):
    contact_numbers.append([])

for step in range(0, 101):
    fileInput = open('../ParticleData/data_%04d' % step, 'r')
    x = []
    y = []
    angle = []
    for line in fileInput:
        numbers = line.split()
        x.append(float(numbers[0]))
        y.append(float(numbers[1]))
        angle.append(float(numbers[2]))

    fileInput = open('index_interior/step_%04d' % step, 'r')
    interior_indices = []
    for line in fileInput:
        interior_indices.append(float(line))
    fileInput.close()

    stats = np.zeros(650)
    matrix = np.zeros([650, 650])
    numbers = []
    fileInput = open('../ParticleData/contact_%04d' % step, 'r')
    for line in fileInput:
        numbers = line.split()
        first = int(float(numbers[0]))
        last = int(float(numbers[1]))
        matrix[first][last] = 1
        matrix[last][first] = 1

    for i in range(0, 650):
        stats[i] = np.sum(matrix[i])
    sum = 0.0
    particle_number = 0.0

    for i in range(0, 650):
        if interior_indices.__contains__(i):
            sum += stats[i]
            particle_number += 1
        else:
            stats[i] = 0

    for i in range(1, 7):
        index = np.where(stats == i)[0]
        for j in range(len(index)):
            I1 = x[index[j]]
            I2 = y[index[j]]
            A0 = angle[index[j]]
            plt.plot([I2 + r0 * m.cos(A0 + 0 * 2 * m.pi / 7), I2 + r0 * m.cos(A0 + 1 * 2 * m.pi / 7),
                      I2 + r0 * m.cos(A0 + 2 * 2 * m.pi / 7), I2 + r0 * m.cos(A0 + 3 * 2 * m.pi / 7),
                      I2 + r0 * m.cos(A0 + 4 * 2 * m.pi / 7), I2 + r0 * m.cos(A0 + 5 * 2 * m.pi / 7),
                      I2 + r0 * m.cos(A0 + 6 * 2 * m.pi / 7), I2 + r0 * m.cos(A0 + 7 * 2 * m.pi / 7)],
                     [I1 + r0 * m.sin(A0 + 0 * 2 * m.pi / 7), I1 + r0 * m.sin(A0 + 1 * 2 * m.pi / 7),
                      I1 + r0 * m.sin(A0 + 2 * 2 * m.pi / 7), I1 + r0 * m.sin(A0 + 3 * 2 * m.pi / 7),
                      I1 + r0 * m.sin(A0 + 4 * 2 * m.pi / 7), I1 + r0 * m.sin(A0 + 5 * 2 * m.pi / 7),
                      I1 + r0 * m.sin(A0 + 6 * 2 * m.pi / 7), I1 + r0 * m.sin(A0 + 7 * 2 * m.pi / 7)], linewidth=1.5,
                     color=s_m.to_rgba(i))
    index = np.where(stats == 0)[0]
    for j in range(len(index)):
        if interior_indices.__contains__(index[j]):
            I1 = x[index[j]]
            I2 = y[index[j]]
            A0 = angle[index[j]]
            plt.plot([I2 + r0 * m.cos(A0 + 0 * 2 * m.pi / 7), I2 + r0 * m.cos(A0 + 1 * 2 * m.pi / 7),
                      I2 + r0 * m.cos(A0 + 2 * 2 * m.pi / 7), I2 + r0 * m.cos(A0 + 3 * 2 * m.pi / 7),
                      I2 + r0 * m.cos(A0 + 4 * 2 * m.pi / 7), I2 + r0 * m.cos(A0 + 5 * 2 * m.pi / 7),
                      I2 + r0 * m.cos(A0 + 6 * 2 * m.pi / 7), I2 + r0 * m.cos(A0 + 7 * 2 * m.pi / 7)],
                     [I1 + r0 * m.sin(A0 + 0 * 2 * m.pi / 7), I1 + r0 * m.sin(A0 + 1 * 2 * m.pi / 7),
                      I1 + r0 * m.sin(A0 + 2 * 2 * m.pi / 7), I1 + r0 * m.sin(A0 + 3 * 2 * m.pi / 7),
                      I1 + r0 * m.sin(A0 + 4 * 2 * m.pi / 7), I1 + r0 * m.sin(A0 + 5 * 2 * m.pi / 7),
                      I1 + r0 * m.sin(A0 + 6 * 2 * m.pi / 7), I1 + r0 * m.sin(A0 + 7 * 2 * m.pi / 7)], linewidth=1.5,
                     color=s_m.to_rgba(0))
    plt.colorbar(s_m, ticks=parameters)
    plt.title('step %03d' % step)
    plt.ylim([0, 3500])
    plt.xlim([0, 3500])
    plt.axis('off')
    plt.savefig('images/step_%03d' % step, dpi=512)
    plt.close()