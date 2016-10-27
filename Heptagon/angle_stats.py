import numpy as np
import math as m
from scipy import misc
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import cv2

def norm(val):
    if -np.pi < val <= np.pi:
        return val
    if val > np.pi:
        return norm(val - 2 * np.pi)
    if val <= -np.pi:
        return norm(val + 2 * np.pi)

os.system('mkdir angle')
r0 = np.load('../tmp/r0.npy')

for step in range(0, 101):
    print step
    contact = np.zeros([620, 620])
    angle = []
    fileInput = open('../ParticleData/data_%04d' % step, 'r')
    cen_x = []
    cen_y = []
    orien = []
    for line in fileInput:
        numbers = line.split()
        cen_y.append(float(numbers[0]))
        cen_x.append(float(numbers[1]))
        orien.append(float(numbers[2]))

    fileInput = open('../ParticleData/contact_%04d' % step, 'r')
    for line in fileInput:
        numbers = line.split()
        first = int(float(numbers[0]))
        last = int(float(numbers[1]))

        if contact[first][last] == 0:
            contact[first][last] = contact[last][first] = 1

            # direction 1
            minVal = 100000
            for i in range(7):
                p_x = cen_x[first] + r0 * m.cos(orien[first] + i * 2 * m.pi / 7)
                p_y = cen_y[first] + r0 * m.sin(orien[first] + i * 2 * m.pi / 7)
                dist2 = (p_x - cen_x[last]) ** 2 + (p_y - cen_y[last]) ** 2
                if dist2 < minVal:
                    minVal = dist2
                    vertex_x = p_x
                    vertex_y = p_y
            phi_1 = m.atan2(vertex_y - cen_y[first], vertex_x - cen_x[first])

            # direction 2
            minVal = 100000
            for i in range(7):
                q_x = cen_x[last] + r0 * m.cos(orien[last] + i * 2 * m.pi / 7)
                q_y = cen_y[last] + r0 * m.sin(orien[last] + i * 2 * m.pi / 7)
                dist2 = (q_x - cen_x[first]) ** 2 + (q_y - cen_y[first]) ** 2
                if dist2 < minVal:
                    minVal = dist2
                    vertex_x = q_x
                    vertex_y = q_y
            phi_2 = m.atan2(vertex_y - cen_y[last], vertex_x - cen_x[last])
            temp = np.abs(norm(phi_1 + m.pi - phi_2))
            if temp > m.pi / 7:
                temp = 2 * m.pi / 7 - temp
            angle.append(temp)
    np.save('angle/step_%04d' % step, angle)

