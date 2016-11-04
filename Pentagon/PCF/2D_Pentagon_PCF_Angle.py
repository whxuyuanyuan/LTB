import numpy as np
import matplotlib.pyplot as plt
from utilities import *
from pc_angle import pairCorrelationFunction_2D
import cv2
from pylab import ginput

# Load the radius of pentagon
r0 = np.load('r0.npy') * 0.8

# Read file
name = '0002'
fileInput = open('data_' + name, 'r')
y, x, a = readFile(name)


# Particle setup
domain_row = 2734
domain_col = 3921
num_particles = x.shape[0]

# Calculation setup
dr = r0 / 15
da = 1
particle_radius = r0
rMax = domain_row / 3

# Compute pair correlation
g_r, r, reference_indices = pairCorrelationFunction_2D(x, y, a, domain_row, domain_col, rMax, dr, da)

# Visualize
plt.figure()
plt.plot(r, g_r[2], color='black')
plt.xlabel('r')
plt.ylabel('g(r)')
plt.xlim( (0, rMax) )
#plt.ylim( (0, 1.05 * g_r.max()) )

# plot_adsorbed_circles(x, y, particle_radius, domain_row, domain_col, reference_indices=reference_indices)
plt.show()











