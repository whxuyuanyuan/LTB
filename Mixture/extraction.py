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
from multiprocessing import Pool


def InsidePolygon(x, y, points):
    """
    Return True if a coordinate (x, y) is inside a polygon defined by a list of verticies [(x1, y1), (x2, x2), ... , (xN, yN)].
    :param x:
    :param y:
    :param points:
    :return:
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


def constructPentMat(r):
    """
    Construct a solid pentagon.
    """
    Mat = np.zeros([2 * int(r * 1.), 2 * int(r * 1.)])
    q1 = r + np.array([r * m.cos(0 * 2 * m.pi / 5), r * m.sin(0 * 2 * m.pi / 5)])
    q2 = r + np.array([r * m.cos(1 * 2 * m.pi / 5), r * m.sin(1 * 2 * m.pi / 5)])
    q3 = r + np.array([r * m.cos(2 * 2 * m.pi / 5), r * m.sin(2 * 2 * m.pi / 5)])
    q4 = r + np.array([r * m.cos(3 * 2 * m.pi / 5), r * m.sin(3 * 2 * m.pi / 5)])
    q5 = r + np.array([r * m.cos(4 * 2 * m.pi / 5), r * m.sin(4 * 2 * m.pi / 5)])

    for it0 in range(0, Mat.shape[0]):
        for it1 in range(0, Mat.shape[1]):
            if InsidePolygon(it1 + 1, it0 + 1, [q1, q2, q3, q4, q5]):
                Mat[it0, it1] = 1

    return Mat

def constructHeptMat(r):
    """
    Construct a solid heptagon.
    """
    Mat = np.zeros([2 * int(r * 1.), 2 * int(r * 1.)])
    q1 = r + np.array([r * m.cos(0 * 2 * m.pi / 7), r * m.sin(0 * 2 * m.pi / 7)])
    q2 = r + np.array([r * m.cos(1 * 2 * m.pi / 7), r * m.sin(1 * 2 * m.pi / 7)])
    q3 = r + np.array([r * m.cos(2 * 2 * m.pi / 7), r * m.sin(2 * 2 * m.pi / 7)])
    q4 = r + np.array([r * m.cos(3 * 2 * m.pi / 7), r * m.sin(3 * 2 * m.pi / 7)])
    q5 = r + np.array([r * m.cos(4 * 2 * m.pi / 7), r * m.sin(4 * 2 * m.pi / 7)])
    q6 = r + np.array([r * m.cos(5 * 2 * m.pi / 7), r * m.sin(5 * 2 * m.pi / 7)])
    q7 = r + np.array([r * m.cos(6 * 2 * m.pi / 7), r * m.sin(6 * 2 * m.pi / 7)])

    for it0 in range(0, Mat.shape[0]):
        for it1 in range(0, Mat.shape[1]):
            if InsidePolygon(it1 + 1, it0 + 1, [q1, q2, q3, q4, q5, q6, q7]):
                Mat[it0, it1] = 1

    return Mat

def resize(it):
    """
    Crop function.
    """
    global I1m, I1M, I2m, I2M
    os.system('convert pictures/' + '%04d' % it + '_Wh.jpg -crop ' + str(I2M - I2m) + 'x' + str(I1M - I1m) + '+' + str(
        I2m) + '+' + str(I1m) + ' pictures/' + '%04d' % it + '_Wh.jpg > /dev/null')
    os.system('convert pictures/' + '%04d' % it + '_Pl.jpg -crop ' + str(I2M - I2m) + 'x' + str(I1M - I1m) + '+' + str(
        I2m) + '+' + str(I1m) + ' pictures/' + '%04d' % it + '_Pl.jpg > /dev/null')
    os.system('convert pictures/' + '%04d' % it + '_Uv.jpg -crop ' + str(I2M - I2m) + 'x' + str(I1M - I1m) + '+' + str(
        I2m) + '+' + str(I1m) + ' pictures/' + '%04d' % it + '_Uv.jpg > /dev/null')


# set parameters
print('Enter first index:')
firstIndex = int(raw_input())
print('Enter last index:')
lastIndex = int(raw_input())
print('Enter number of parallel processing:')
Npara = int(raw_input())

# rename pictures
for i in range(0, (lastIndex - firstIndex)/3 + 1):
    os.system('mv pictures/IMG_' + '%04d' % (firstIndex + 3 * i) + '.JPG pictures/' + '%04d' % i + '_Uv.jpg')
    os.system('mv pictures/IMG_' + '%04d' % (firstIndex + 3 * i + 1) + '.JPG pictures/' + '%04d' % i + '_Wh.jpg')
    os.system('mv pictures/IMG_' + '%04d' % (firstIndex + 3 * i + 2) + '.JPG pictures/' + '%04d' % i + '_Pl.jpg')

# select extremum points
imgWh = misc.imread('pictures/0000_Wh.jpg');
plt.imshow(imgWh)
plt.title('select 2 horizontal and vertical extremum points, then middle click and close')
p1, p2, p3, p4 = ginput(0, 0)
plt.show()
plt.close()

# extract limit coordinates:
I1m = int(p3[1])
I1M = int(p4[1])
I2m = int(p1[0])
I2M = int(p2[0])

# resize of the pictures:
p = Pool(8)
p.map(resize, range(0, (lastIndex - firstIndex) / 3 + 2))

# make movies
os.system('mkdir movies')
os.system(
    'ffmpeg -r 10 -f image2 -i pictures/%04d_Wh.jpg -qscale 1 -vf scale=1500:1000 movies/MovieWh.avi >/dev/null 2>&1')
os.system(
    'ffmpeg -r 10 -f image2 -i pictures/%04d_Pl.jpg -qscale 1 -vf scale=1500:1000 movies/MoviePl.avi >/dev/null 2>&1')
os.system(
    'ffmpeg -r 10 -f image2 -i pictures/%04d_Uv.jpg -qscale 1 -vf scale=1500:1000 movies/MovieUv.avi >/dev/null 2>&1')


# extract the radius of circumcircle
# Pentagon
plt.imshow(imgWh)
plt.title('zoom and click on 5 vertices of a pentagon, then middle click and close')
_, q1, q2, q3, q4, q5 = ginput(0, 0)
plt.show()
plt.close()

r0_Pent = int((0.5 / m.sin(m.pi / 5)) * np.mean(np.array(
    [m.sqrt((q1[0] - q2[0]) ** 2 + (q1[1] - q2[1]) ** 2), m.sqrt((q2[0] - q3[0]) ** 2 + (q2[1] - q3[1]) ** 2),
     m.sqrt((q3[0] - q4[0]) ** 2 + (q3[1] - q4[1]) ** 2), m.sqrt((q4[0] - q5[0]) ** 2 + (q4[1] - q5[1]) ** 2),
     m.sqrt((q5[0] - q1[0]) ** 2 + (q5[1] - q1[1]) ** 2)])))

# Heptagon
plt.imshow(imgWh)
plt.title('zoom and click on 7 vertices of a heptagon, then middle click and close')
_, q1, q2, q3, q4, q5, q6, q7 = ginput(0, 0)
plt.show()
plt.close()

r0_Hept = (0.5 / m.sin(m.pi / 7)) * np.mean(np.array(
    [m.sqrt((q1[0] - q2[0]) ** 2 + (q1[1] - q2[1]) ** 2), m.sqrt((q2[0] - q3[0]) ** 2 + (q2[1] - q3[1]) ** 2),
     m.sqrt((q3[0] - q4[0]) ** 2 + (q3[1] - q4[1]) ** 2), m.sqrt((q4[0] - q5[0]) ** 2 + (q4[1] - q5[1]) ** 2),
     m.sqrt((q5[0] - q6[0]) ** 2 + (q5[1] - q6[1]) ** 2), m.sqrt((q6[0] - q7[0]) ** 2 + (q6[1] - q7[1]) ** 2),
     m.sqrt((q7[0] - q1[0]) ** 2 + (q7[1] - q1[1]) ** 2)]))

# construct a normal matrix
MatPent = constructPentMat(r0_Pent)
MatHept = constructHeptMat(r0_Hept)

# make the pattern of edges:
MatPentEdge = cv2.Canny(1 - MatPent.astype('uint8'), 0, 1)
kernel = np.ones((int(MatPentEdge.shape[0] * 0.008) * 2 + 1, int(MatPentEdge.shape[0] * 0.008) * 2 + 1), np.uint8)
MatPentEdge = cv2.dilate(MatPentEdge, kernel, iterations=1)

MatHeptEdge = cv2.Canny(1 - MatHept.astype('uint8'), 0, 1)
kernel = np.ones((int(MatHeptEdge.shape[0] * 0.008) * 2 + 1, int(MatHeptEdge.shape[0] * 0.008) * 2 + 1), np.uint8)
MatHeptEdge = cv2.dilate(MatHeptEdge, kernel, iterations=1)

# construct a smaller matrix
r1_Pent = 0.97 * r0_Pent
MatPentS = constructPentMat(r1_Pent)

r1_Hept = 0.97 * r0_Hept
MatHeptS = constructHeptMat(r1_Hept)

# construct a larger matrix
r2_Pent = 1.1 * r0_Pent
MatPentL = constructPentMat(r2_Pent)

r2_Hept = 1.1 * r0_Hept
MatHeptL = constructHeptMat(r2_Hept)

# make folders to save data:
os.system('mkdir movies/tmpPositionOnImage')
os.system('mkdir movies/tmpPosition')
os.system('mkdir movies/tmpPositionContact')
os.system('mkdir ParticleData')
os.system('mkdir rough_position')

# save global variables
os.system('mkdir tmp')
np.save('tmp/r0_Pent.npy', r0_Pent)
np.save('tmp/r1_Pent.npy', r1_Pent)
np.save('tmp/r2_Pent.npy', r2_Pent)
np.save('tmp/r0_Hept.npy', r0_Hept)
np.save('tmp/r1_Hept.npy', r1_Hept)
np.save('tmp/r2_Hept.npy', r2_Hept)
np.save('tmp/MatPent.npy', MatPent)
np.save('tmp/MatPentEdge.npy', MatPentEdge)
np.save('tmp/MatPentS.npy', MatPentS)
np.save('tmp/MatPentL.npy', MatPentL)
np.save('tmp/MatHept.npy', MatHept)
np.save('tmp/MatHeptEdge.npy', MatHeptEdge)
np.save('tmp/MatHeptS.npy', MatHeptS)
np.save('tmp/MatHeptL.npy', MatHeptL)

# segmentation vector:
Segment = np.linspace(0, (lastIndex - firstIndex) / 3 + 1, Npara + 1).astype(int)

# segmentation loop:
for itCp in range(1, Npara + 1):
    # Create computation folder:
    os.system('mkdir Compute_' + '%02d' % itCp)
    # Transfer global data:
    os.system('cp -r tmp Compute_' + '%02d' % itCp + '/tmp')
    # Store begining and end:
    np.save('Compute_' + '%02d' % itCp + '/begin.npy', Segment[itCp - 1])
    if itCp < Npara:
        np.save('Compute_' + '%02d' % itCp + '/stop.npy', Segment[itCp] - 1)
    else:
        np.save('Compute_' + '%02d' % itCp + '/stop.npy', Segment[itCp])
    # Transfer main code:
    os.system('cp position.py Compute_' + '%02d' % itCp + '/position.py')
    os.system('cp contact.py Compute_' + '%02d' % itCp + '/contact.py')

print 'Run each code'

