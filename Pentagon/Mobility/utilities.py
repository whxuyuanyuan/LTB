import numpy as np


# read file
def readFile(number):
    fileInput = open('tracking/particle_%03d.txt' % number, 'r')
    x = []
    y = []
    a = []
    for line in fileInput:
        numbers = line.split()
        x.append(float(numbers[1]))
        y.append(float(numbers[2]))
        a.append(float(numbers[3]))
    x = np.array(x)
    y = np.array(y)
    a = np.array(a)
    return x, y, a


# return the distance of two points
def distance(x1, y1, x2, y2):
    dis = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dis


# plot the pentagon
def poltPentagon(I1, I2, A0, r0, color):
    import matplotlib.pyplot as plt
    import math as m
    plt.plot([I2 + r0 * m.cos(A0 + 0 * 2 * m.pi / 5), I2 + r0 * m.cos(A0 + 1 * 2 * m.pi / 5),
                  I2 + r0 * m.cos(A0 + 2 * 2 * m.pi / 5), I2 + r0 * m.cos(A0 + 3 * 2 * m.pi / 5),
                  I2 + r0 * m.cos(A0 + 4 * 2 * m.pi / 5), I2 + r0 * m.cos(A0 + 5 * 2 * m.pi / 5)],
                 [I1 + r0 * m.sin(A0 + 0 * 2 * m.pi / 5), I1 + r0 * m.sin(A0 + 1 * 2 * m.pi / 5),
                  I1 + r0 * m.sin(A0 + 2 * 2 * m.pi / 5), I1 + r0 * m.sin(A0 + 3 * 2 * m.pi / 5),
                  I1 + r0 * m.sin(A0 + 4 * 2 * m.pi / 5), I1 + r0 * m.sin(A0 + 5 * 2 * m.pi / 5)],
                 linewidth=1.5, color='%s' % color)