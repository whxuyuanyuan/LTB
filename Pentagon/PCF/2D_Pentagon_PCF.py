import numpy as np
import matplotlib.pyplot as plt
from utilities import *
from paircorrelation import pairCorrelationFunction_2D
import cv2
from pylab import ginput

# Load the radius of pentagon
r0 = np.load('r0.npy') * 0.8

for i in range(2,4,2):
    name = '%04d'%(i)
    y, x, a = readFile(name)


    # Particle setup
    domain_row = 3000
    domain_col = 3900
    num_particles = x.shape[0]

    # Calculation setup
    dr = r0 / 15

    particle_radius = r0
    rMax = domain_row / 3

    # Compute pair correlation
    g_r, r, reference_indices = pairCorrelationFunction_2D(x, y, domain_row, domain_col, rMax, dr)

    # Visualize
    plt.figure()
    plt.plot(r, g_r, color='black')
    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.xlim( (0, rMax) )
    plt.ylim( (0, 6.5) )
    name = '%04d'%(i/2)
    plt.title('step ' + name)
    plt.savefig('pcf/step_' + name + '.png', dpi=250)

    plot_adsorbed_circles(x, y, particle_radius, domain_row, domain_col, reference_indices=reference_indices)
    plt.show()











