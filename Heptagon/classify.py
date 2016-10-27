# Load usefull libraries:
import numpy as np
import math as m
from scipy import misc
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import cv2


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


class MyContact:
    def __init__(self, x, y, isFlat):
        self.x = x
        self.y = y
        self.isFlat = isFlat

def norm(val):
    if -np.pi < val <= np.pi:
        return val
    if val > np.pi:
        return norm(val - 2 * np.pi)
    if val <= -np.pi:
        return norm(val + 2 * np.pi)

os.system('mkdir classification')

r0 = np.load('../tmp/r0.npy')

flat = []
point = []

for step in range(0, 101):
    print step
    fileInput = open('../ParticleData/data_%04d' % step, 'r')
    cen_x = []
    cen_y = []
    angle = []
    for line in fileInput:
        numbers = line.split()
        cen_y.append(float(numbers[0]))
        cen_x.append(float(numbers[1]))
        angle.append(float(numbers[2]))

    flat_matrix = np.zeros([650, 650])
    point_matrix = np.zeros([650, 650])

    imgWh = misc.imread('../pictures/%04d_Wh.jpg' % step)
    gray = rgb2gray(imgWh)
    imgWhBin = (gray < 175).astype('int')

    numbers = []
    fileInput = open('../ParticleData/contact_%04d' % step, 'r')
    for line in fileInput:
        numbers = line.split()
        first = int(float(numbers[0]))
        last = int(float(numbers[1]))
        x_temp = float(numbers[3])
        y_temp = float(numbers[2])
        isFlat = False

        subMap = imgWhBin[int(y_temp) - 20: int(y_temp) + 21, int(x_temp) - 20: int(x_temp) + 21]
        if np.sum(subMap) == 41 * 41:
            isFlat = True

        if not isFlat:
            phi_dir1 = angle[first]
            phi_con1 = m.atan2(y_temp - cen_y[first], x_temp - cen_x[first])
            phi_normal1 = 0
            if phi_dir1 < phi_con1:
                theta_1 = phi_dir1
                theta_2 = 0
                while theta_1 + 2 * m.pi / 7 < phi_con1:
                    theta_1 += 2 * m.pi / 7
                theta_2 = theta_1 + 2 * m.pi / 7
                phi_normal1 = norm((theta_1 + theta_2) * 0.5)
            else:
                theta_1 = phi_dir1
                theta_2 = 0
                while theta_1 - 2 * m.pi / 7 > phi_con1:
                    theta_1 -= 2 * m.pi / 7
                theta_2 = theta_1 - 2 * m.pi / 7
                phi_normal1 = norm((theta_1 + theta_2) * 0.5)

            phi_dir2 = angle[last]
            phi_con2 = m.atan2(y_temp - cen_y[last], x_temp - cen_x[last])
            phi_normal2 = 0
            if phi_dir2 < phi_con2:
                theta_1 = phi_dir2
                theta_2 = 0
                while theta_1 + 2 * m.pi / 7 < phi_con2:
                    theta_1 += 2 * m.pi / 7
                theta_2 = theta_1 + 2 * m.pi / 7
                phi_normal2 = norm((theta_1 + theta_2) * 0.5)
            else:
                theta_1 = phi_dir2
                theta_2 = 0
                while theta_1 - 2 * m.pi / 7 > phi_con2:
                    theta_1 -= 2 * m.pi / 7
                theta_2 = theta_1 - 2 * m.pi / 7
                phi_normal2 = norm((theta_1 + theta_2) * 0.5)

            if np.abs(norm(phi_normal1 - phi_normal2 + np.pi)) < 0.06:
                isFlat = True

        if isFlat:
            #plt.scatter(x_temp, y_temp, facecolor='r', linewidths=0.1, s=2)
            flat_matrix[first][last] = flat_matrix[last][first] = 1.0
        else:
            #plt.scatter(x_temp, y_temp, facecolor='b', linewidths=0.1, s=2)
            point_matrix[first][last] = point_matrix[last][first] = 1.0
    flat.append(np.sum(flat_matrix))
    point.append(np.sum(point_matrix))


np.save('flat', flat)
np.save('point', point)

axis1 = 38.60
axis2 = 38.32
stepsize = 0.04
particleArea = 618.0 * 7.0 / 4.0 * 0.6934**2 / np.tan(np.pi / 7)
phi = [particleArea / (axis1 * axis2)]

for i in range(1, 51):
    axis1 -= stepsize
    axis2 -= stepsize
    phi.append(particleArea / (axis1 * axis2))

for i in range(49, -1, -1):
    phi.append(phi[i])


plt.plot(phi[0: 51], flat[0: 51], 'b.-', label='Flat, Compression')
plt.plot(phi[50: 100], flat[50: 100], 'bx-', label='Flat, Deompression')

plt.plot(phi[0: 51], point[0: 51], 'g.-', label='Point, Compression')
plt.plot(phi[50: 100], point[50: 100], 'gx-', label='point, Decompression')

plt.xlabel('Packing fraction')
plt.legend(loc='upper left')
plt.savefig('classification.eps')
plt.show()
