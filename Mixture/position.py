import numpy as np
import math as m
from scipy import misc
from scipy import optimize
from scipy import stats
from scipy import ndimage as nd
from pylab import ginput
import cv2
import cv2.cv as cv
import os

def norm(val):
    """
    Return a value in the range of -pi and pi.
    """
    if -np.pi < val <= np.pi:
        return val
    if val > np.pi:
        return norm(val - 2 * np.pi)
    if val <= -np.pi:
        return norm(val + 2 * np.pi)

def InsidePolygon(x, y, points):
    """
    Return True if a coordinate (x, y) is inside a polygon defined by a list of verticies [(x1, y1), (x2, x2), ... , (xN, yN)].
    """
    n = len(points)
    inside = False
    p1x, p1y = points[0]
    for i in range(1, n + 1):
        p2x, p2y = points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if (p1x == p2x) or (x <= xinters):
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside



def OverlapVal_Pent(x):
    # Define global variables:
    global MatPent, MatPentEdge, subMap1, subMap2, r0_Pent, angle
    # Extract data:
    Ia = int(x[0])
    Ib = int(x[1])
    A0 = x[2]
    if (abs(Ia) < int(0.1 * r0_Pent)) and (abs(Ib) < int(0.1 * r0_Pent)):
        try:
            # Rotate pattern matrices:
            MatPentRot = (nd.interpolation.rotate(MatPent, A0, reshape=False, cval=0.0) > 0.2).astype(float)
            MatPentEdgeRot = (nd.interpolation.rotate(MatPentEdge, A0, reshape=False, cval=0.0) > 0.2).astype(float)
            # Compute overlap value:
            Val0a = (np.sum(MatPentRot * subMap1[
                                         subMap1.shape[0] / 2 + Ia - MatPentRot.shape[0] / 2:subMap1.shape[0] / 2 + Ia +
                                                                                           MatPentRot.shape[0] / 2,
                                         subMap1.shape[1] / 2 + Ib - MatPentRot.shape[1] / 2:subMap1.shape[1] / 2 + Ib +
                                                                                           MatPentRot.shape[
                                                                                               1] / 2]) / np.sum(
                MatPentRot))
            Val0b = (np.sum(MatPentEdgeRot * subMap2[subMap2.shape[0] / 2 + Ia - MatPentEdgeRot.shape[0] / 2:subMap2.shape[
                                                                                                             0] / 2 + Ia +
                                                                                                         MatPentEdgeRot.shape[
                                                                                                             0] / 2,
                                             subMap2.shape[1] / 2 + Ib - MatPentEdgeRot.shape[1] / 2:subMap2.shape[
                                                                                                       1] / 2 + Ib +
                                                                                                   MatPentEdgeRot.shape[
                                                                                                       1] / 2]) / np.sum(
                MatPentEdgeRot))
            Val0a = np.amax(np.array([0.1, Val0a]))
            Val0b = np.amax(np.array([0.1, Val0b]))
            if Val0a < 0.5:
                Val0 = 1 - 0.1 / Val0a - 0.9 / Val0b
            else:
                Val0 = 0.5 / Val0a + 0.5 / Val0b
        except:
            Val0 = 100
    else:
        Val0 = 100 + np.amax([abs(Ia) - int(0.1 * r0_Pent), abs(Ib) - int(0.1 * r0_Pent)])
    return Val0


def OverlapVal_Hept(x):
    global MatHept, MatHeptEdge, subMap1, subMap2, r0_Hept, angle
    dx = x[0]
    dy = x[1]
    dA = x[2]
    A = angle + dA
    val = 100
    if (abs(dx) < int(0.05 * r0_Hept)) and (abs(dy) < int(0.05 * r0_Hept) and abs(dA) < 0.05):
        try:
            # Rotate pattern matrix
            MatHeptRot = (nd.interpolation.rotate(MatHept, A/m.pi*180, reshape=False, cval=0.0) > 0.2).astype(float)
            MatHeptEdgeRot = (nd.interpolation.rotate(MatHeptEdge, A/m.pi*180, reshape=False, cval=0.0) > 0.2).astype(float)
            # Compute overlap value

            val1 = (np.sum(MatHeptRot * subMap1[
                                       subMap1.shape[0] / 2 + dx - MatHeptRot.shape[0] / 2:subMap1.shape[0] / 2 + dx +
                                                                                          MatHeptRot.shape[0] / 2,
                                       subMap1.shape[1] / 2 + dy - MatHeptRot.shape[1] / 2:subMap1.shape[1] / 2 + dy +
                                                                                          MatHeptRot.shape[
                                                                                               1] / 2]) / np.sum(
                MatHeptRot))

            val2 = (np.sum(MatHeptEdgeRot * subMap2[subMap2.shape[0] / 2 + dx - MatHeptEdgeRot.shape[0] /
                                                                                2:subMap2.shape[0] / 2 + dx +
                                                                                  MatHeptEdgeRot.shape[0] / 2,
                                            subMap2.shape[1] / 2 + dy - MatHeptEdgeRot.shape[1] / 2:subMap2.shape[1] / 2
                                                                                                    + dy +
                                                                                                    MatHeptEdgeRot.shape[1]
                                                                                                    / 2]) /
                    np.sum(MatHeptEdgeRot))
            val = 0.4/val1 + 0.6/val2
        except:
            val = 100
    return val

# Load params
r0_Pent = np.load('tmp/r0_Pent.npy')    # radius of pentagon
MatPent = np.load('tmp/MatPent.npy')    # solid pentagon
MatPentEdge = np.load('tmp/MatPentEdge.npy')    # hollow pentagon
r0_Hept = np.load('tmp/r0_Hept.npy')    # radius of heptagon
MatHept = np.load('tmp/MatHept.npy')    # solid heptagon
MatHeptEdge = np.load('tmp/MatHeptEdge.npy')    # hollow heptagon
begin = np.load('begin.npy')
stop = np.load('stop.npy')

for step in range(begin, stop + 1):
    print step

    # Load images
    imgUv = misc.imread('../pictures/%04d_Uv.jpg' % step)
    imgWh = misc.imread('../pictures/%04d_Wh.jpg' % step)
    imgWh = imgWh[:, :, 0]  # extract red part

    pent_cen_x = []
    pent_cen_y = []
    pent_orien = []
    hept_cen_x = []
    hept_cen_y = []
    hept_orien = []

    f = open('../rough_position/step_%04d' % step, 'r')
    for line in f:
        data = [float(elem) for elem in line.split()]
        if len(data) > 2:
            hept_cen_y.append(data[0])
            hept_cen_x.append(data[1])
            hept_orien.append(data[2])
        else:
            pent_cen_y.append(data[0])
            pent_cen_x.append(data[1])

    print len(pent_cen_x), len(hept_cen_x)

    for i in range(len(pent_cen_x)):
        x1 = int(pent_cen_x[i])
        y1 = int(pent_cen_y[i])
        # Extract interesting sub-part of the matrix
        subMap = imgWh[x1 - int(1.1 * r0_Pent):x1 + int(1.1 * r0_Pent), y1 - int(1.1 * r0_Pent):y1 + int(1.1 * r0_Pent)]
        # Threshold the the sub-matrix:
        # subMap = cv2.GaussianBlur(subMap.astype('uint8'),
        #                       (int(subMap.shape[0] * 0.02 / 2) * 2 + 1, int(subMap.shape[0] * 0.02 / 2) * 2 + 1), 0)
        subMap = subMap.astype('uint8')
        subMap1 = (subMap < 180).astype(int) * (subMap > 70).astype(int)
        subMap2 = cv2.adaptiveThreshold(subMap, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
	subMap2 = (subMap2 < 120).astype(int)
        count = 3
        Val1 = 2
        while (Val1 > 1.5) and (count > 0):
            x0 = [0, 0, count * 20.]
            optim = optimize.minimize(OverlapVal_Pent, x0, method='Powell',
                                      options={'disp': False, 'maxfev': 150, 'ftol': 10 ** (-5), 'xtol': 10 ** (-5)})
            Val1 = optim.fun
            if count == 3:
                StoreDat = np.array([optim.x[0], optim.x[1], optim.x[2], Val1])
            else:
                StoreDat = np.vstack((StoreDat, np.array([optim.x[0], optim.x[1], optim.x[2], Val1])))
            count = count - 1
        # Store data:
        if count == 2:
            pent_cen_x[i] = x1 + StoreDat[0];
            pent_cen_y[i] = y1 + StoreDat[1];
            pent_orien.append(norm(-StoreDat[2] * m.pi / 180.0));
        else:
            ValTot = StoreDat[:, 3]
            I = np.where(ValTot == np.amin(ValTot))[0][0]
            pent_cen_x[i] = x1 + StoreDat[I, 0];
            pent_cen_y[i] = y1 + StoreDat[I, 1];
            pent_orien.append(norm(-StoreDat[I, 2] * m.pi / 180.0));

    for i in range(len(hept_cen_x)):
        x1 = hept_cen_x[i]
        y1 = hept_cen_y[i]
        angle = hept_orien[i]
        # Extract interesting sub-part of the matrix
        subMap = imgWh[x1 - int(1.1 * r0_Hept):x1 + int(1.1 * r0_Hept), y1 - int(1.1 * r0_Hept):y1 + int(1.1 * r0_Hept)]
        # Threshold the the sub-matrix:
        # subMap = cv2.GaussianBlur(subMap.astype('uint8'),
        #                       (int(subMap.shape[0] * 0.02 / 2) * 2 + 1, int(subMap.shape[0] * 0.02 / 2) * 2 + 1), 0)
        subMap = subMap.astype('uint8')
        subMap1 = (subMap < 180).astype(int) * (subMap > 100).astype(int)
        subMap2 = cv2.adaptiveThreshold(subMap, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        subMap2 = (subMap2 < 120).astype(int)
        guess = [0.0, 0.0, 0.0]
        optim = optimize.minimize(OverlapVal_Hept, guess, method='Powell',
                                  options={'disp': False, 'maxfev': 150, 'ftol': 10 ** (-5), 'xtol': 10 ** (-5)})
        hept_cen_x[i] += optim.x[0]
        hept_cen_y[i] += optim.x[1]
        hept_orien[i] += optim.x[2]

    # save data
    data = np.transpose(np.array([pent_cen_x, pent_cen_y, pent_orien]))
    np.savetxt('../ParticleData/data_pent_'+'%04d' % step, data, delimiter=' ', newline='\n')

    data = np.transpose(np.array([hept_cen_x, hept_cen_y, hept_orien]))
    np.savetxt('../ParticleData/data_hept_'+'%04d' % step, data, delimiter=' ', newline='\n')


