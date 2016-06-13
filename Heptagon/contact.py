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


############################### Global Variables ###############################
global MatHept, MatHeptEdge, Map2a, Map2b, r0


############################### Load Data ######################################
r0 = np.load('tmp/r0.npy')  # normal radius
r1 = np.load('tmp/r1.npy')  # small radius
r2 = np.load('tmp/r2.npy')  # large radius
MatHept = np.load('tmp/MatHept.npy')    # solid heptagon
MatHeptEdge = np.load('tmp/MatHeptEdge.npy') # hollow heptagon
MatHeptS = np.load('tmp/MatHeptS.npy')  # small pattern
MatHeptL = np.load('tmp/MatHeptL.npy')  # large patten
begin = np.load('begin.npy')    # start
stop = np.load('stop.npy')  # end


############################## Functions #######################################
def InsidePolygon(x,y,points):
    '''Return True if a coordinate (x, y) is inside a polygon defined by a list of verticies [(x1, y1), (x2, x2), ... , (xN, yN)].'''
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


############################## Main ############################################
for itQ in range(begin, stop + 1):   
    print str(itQ) 

    # Load image 
    ImgWh = misc.imread('../pictures/' + '%04d' % itQ + '_Wh.jpg')

    # Import data
    fileInput = open('../ParticleData/data_' + '%04d' % itQ)
    VecI1 = []
    VecI2 = []
    VecA = []
    numbers = []
    
    for line in fileInput:
    	numbers = line.split()
    	VecI1.append(float(numbers[0]))
    	VecI2.append(float(numbers[1]))
    	VecA.append(float(numbers[2]))

    VecI1 = np.array(VecI1)
    VecI2 = np.array(VecI2)
    VecA = np.array(VecA) / np.pi * 180.0  # convert degrees to radians

    # Measurement of the pressure
    
    # Load and reduce the polarized picture
    ImgPl = misc.imread('../pictures/' + '%04d' % itQ + '_Pl.jpg')
    ImgPl = ImgPl[:, :, 1]

    # Computation of G^2 for the full map
    # Smoothen picture:
    ImgPl1 = cv2.GaussianBlur(ImgPl.astype('uint8'), (9, 9), 0).astype('float')
    # Compute G2 for each pixel
    ImG2 = np.zeros(ImgPl.shape)
    for it0 in range(0, ImG2.shape[0]):
        for it1 in range(0, ImG2.shape[1]):
            if (it0 > 0) and (it1 > 0) and (it0 < ImgPl1.shape[0] - 1) and (it1 < ImgPl1.shape[1] - 1): 
                ImG2[it0, it1] = 1 / 4. * ((ImgPl1[it0 + 1, it1] - ImgPl1[it0 - 1, it1]) ** 2. + (ImgPl1[it0, it1 + 1] - ImgPl1[it0, it1 - 1]) ** 2. + 1 / 2. *(ImgPl1[it0 + 1, it1 + 1] - ImgPl1[it0 - 1, it1 - 1]) ** 2. + 1 / 2. * (ImgPl1[it0 + 1, it1 - 1] - ImgPl1[it0 - 1, it1 + 1]) ** 2.)
    
    # Measurement of G2 for each particle:
    # Loop over particles:
    VecG2 = np.zeros(len(VecI1));
    for itPt in range(0, len(VecI1)):
        # Load properties of current particle:
        I1c = VecI1[itPt]
        I2c = VecI2[itPt]
        Ac = VecA[itPt]
        # Rotate pattern:
        MatHeptSRot=(nd.interpolation.rotate(MatHeptS,Ac,reshape=False,cval=0.0)>0.2).astype(float)
        # Compute integral:
        VecG2[itPt] = np.sum(ImG2[int(I1c - MatHeptS.shape[0] / 2): int(I1c - MatHeptS.shape[0] / 2) + MatHeptS.shape[0], int(I2c - MatHeptS.shape[1] / 2): int(I2c - MatHeptS.shape[1] / 2) + MatHeptS.shape[1]] * MatHeptSRot)

    
    # Measurement of the contacts:
    
    # Loop over particles:
    VecCont = np.array([0, 0, 0, 0])
    ImgPl2 = stats.threshold(ImgPl1, threshmin=40, threshmax=200, newval=0)
    for itPt in range(0, len(VecI1)):
        # Load properties of current particle:
        I1c = VecI1[itPt]
        I2c = VecI2[itPt]
        Ac = VecA[itPt]
        # Detect neighbors:
        d0 = np.sqrt((I1c - VecI1) ** 2 + (I2c - VecI2) ** 2)
        In = np.where((d0 < 2 * r2) & (d0 > 0))[0]
        if len(In) > 0:
            # Rotate main patterns:
            MatHeptLRot0 = (nd.interpolation.rotate(MatHeptL, Ac, reshape=False, cval=0.0) > 0.2).astype(float)
            MatHeptRot0 = (nd.interpolation.rotate(MatHept, Ac, reshape=False, cval=0.0) > 0.2).astype(float)
            # Loop over neighbors:
            for itPtb in range(0,len(In)):
                # Load properties of neighbors particle:
                I1d = VecI1[In[itPtb]]
                I2d = VecI2[In[itPtb]]
                Ad = VecA[In[itPtb]]
                # Rotate the neighbor patterns:
                MatHeptLRot1 = (nd.interpolation.rotate(MatHeptL, Ad, reshape=False, cval=0.0) > 0.2).astype(float)
                MatHeptRot1 = (nd.interpolation.rotate(MatHept, Ad, reshape=False, cval=0.0) > 0.2).astype(float)
                # Make the overlapping map:
                ImgCont = np.zeros(ImgPl2.shape);

        
		try:
            ImgCont[int(I1c - MatHeptL.shape[0] / 2): int(I1c - MatHeptL.shape[0] / 2) + MatHeptL.shape[0],int(I2c - MatHeptL.shape[1] / 2): int(I2c - MatHeptL.shape[1] / 2) + MatHeptL.shape[1]] = ImgCont[int(I1c - MatHeptL.shape[0] / 2): int(I1c - MatHeptL.shape[0] / 2) + MatHeptL.shape[0], int(I2c - MatHeptL.shape[1] / 2): int(I2c - MatHeptL.shape[1] / 2) + MatHeptL.shape[1]] + MatHeptLRot0
		except:
			temp = ImgCont[int(I1c - MatHeptL.shape[0] / 2): int(I1c - MatHeptL.shape[0] / 2) + MatHeptL.shape[0], int(I2c - MatHeptL.shape[1] / 2): int(I2c - MatHeptL.shape[1] / 2) + MatHeptL.shape[1]]
			ImgCont[int(I1c - MatHeptL.shape[0] / 2): int(I1c - MatHeptL.shape[0] / 2) + MatHeptL.shape[0], int(I2c - MatHeptL.shape[1] / 2): int(I2c - MatHeptL.shape[1] / 2) + MatHeptL.shape[1]] = ImgCont[int(I1c - MatHeptL.shape[0] / 2): int(I1c - MatHeptL.shape[0] / 2) + MatHeptL.shape[0], int(I2c - MatHeptL.shape[1] / 2): int(I2c - MatHeptL.shape[1] / 2) + MatHeptL.shape[1]] + MatHeptLRot0[0: temp.shape[0], 0: temp.shape[1]]
		try:
            ImgCont[int(I1d - MatHeptL.shape[0] / 2): int(I1d - MatHeptL.shape[0] / 2) + MatHeptL.shape[0], int(I2d - MatHeptL.shape[1] / 2): int(I2d - MatHeptL.shape[1] / 2) + MatHeptL.shape[1]] = ImgCont[int(I1d - MatHeptL.shape[0] / 2): int(I1d - MatHeptL.shape[0] / 2) + MatHeptL.shape[0], int(I2d - MatHeptL.shape[1] / 2): int(I2d - MatHeptL.shape[1] / 2) + MatHeptL.shape[1]] + MatHeptLRot1
		except:
			temp = ImgCont[int(I1d - MatHeptL.shape[0] / 2): int(I1d - MatHeptL.shape[0] / 2) + MatHeptL.shape[0], int(I2d - MatHeptL.shape[1] / 2): int(I2d - MatHeptL.shape[1] / 2) + MatHeptL.shape[1]]
			ImgCont[int(I1d - MatHeptL.shape[0] / 2): int(I1d - MatHeptL.shape[0] / 2) + MatHeptL.shape[0], int(I2d - MatHeptL.shape[1] / 2): int(I2d - MatHeptL.shape[1] / 2) + MatHeptL.shape[1]] = ImgCont[int(I1d - MatHeptL.shape[0] / 2): int(I1d - MatHeptL.shape[0] / 2) + MatHeptL.shape[0], int(I2d - MatHeptL.shape[1] / 2): int(I2d - MatHeptL.shape[1] / 2) + MatHeptL.shape[1]] + MatHeptLRot1[0: temp.shape[0], 0: temp.shape[1]]
        

        # If there i overlapping:
        if np.amax(ImgCont) > 1:
            # Expend the overlap:
            ImgCont = (ImgCont > 1).astype('uint8')
            kernel = np.ones((11, 11), np.uint8)
            ImgCont1 = cv2.dilate(ImgCont, kernel, iterations=1)
            # Measure polarization intensity at the overlap:
            ImgTmp = np.zeros(ImgPl2.shape)
            ImgTmp[int(I1c - MatHept.shape[0] / 2): int(I1c - MatHept.shape[0] / 2) + MatHept.shape[0], int(I2c - MatHept.shape[1] / 2): int(I2c - MatHept.shape[1] / 2) + MatHept.shape[1]] = ImgTmp[int(I1c - MatHept.shape[0] / 2): int(I1c - MatHept.shape[0] / 2) + MatHept.shape[0], int(I2c - MatHept.shape[1] / 2): int(I2c - MatHept.shape[1] / 2) + MatHept.shape[1]] + MatHeptRot0
            Val00 = np.sum(ImgCont1*ImgTmp*ImgPl2)
            ImgTmp = np.zeros(ImgPl2.shape)
            ImgTmp[int(I1c - MatHept.shape[0] / 2): int(I1c - MatHept.shape[0] / 2) + MatHept.shape[0], int(I2c - MatHept.shape[1] / 2): int(I2c - MatHept.shape[1] / 2) + MatHept.shape[1]] = ImgTmp[int(I1c - MatHept.shape[0] / 2): int(I1c - MatHept.shape[0] / 2) + MatHept.shape[0], int(I2c - MatHept.shape[1] / 2): int(I2c - MatHept.shape[1] / 2) + MatHept.shape[1]] + MatHeptRot1
            Val01 = np.sum(ImgCont1 * ImgTmp * ImgPl2)
            if (Val00 > 300) and (Val01 > 300):
                ImgTmp = ImgCont1 * ImgPl2
                I1cont, I2cont = np.where(ImgCont > 0)
                I1cont = np.mean(I1cont)
                I2cont = np.mean(I2cont); 
                VecCont = np.vstack((VecCont, np.array([itPt, In[itPtb], I1cont, I2cont])))
    
    VecCont = np.delete(VecCont,0,0)
    
    # Plot
    ImgPl0 = misc.imread('../pictures/' + '%04d' % itQ + '_Pl.jpg')  
    plt.imshow(ImgPl0) 
    np.save('VecCont', VecCont)
    if len(VecCont) > 3:
	    for itCt in range(0,len(VecCont)):
		I1 = VecCont[itCt,2]
        I2 = VecCont[itCt,3]; 
		plt.scatter(I2, I1,facecolor='r', linewidth=0.1, s=3)
    
    plt.xlim([0,ImgPl0.shape[1]])
    plt.ylim([0,ImgPl0.shape[0]])
    plt.axis('equal')
    plt.axis('off')
    plt.title('step '+'%04d'%itQ)
    plt.savefig('../movies/tmpPositionContact/'+'%04d'%itQ+'.png',dpi=250)
    plt.close()

    
    ##Save contacts (|number particle 1 | number particle 2 | contact matrix coordinate 1| contact matrix coordinate 2) :
    np.savetxt('../ParticleData/contact_'+'%04d'%itQ,VecCont,delimiter=' ',newline='\n')

















