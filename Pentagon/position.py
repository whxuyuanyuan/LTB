#Load usefull libraries:
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


#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#
#Define global variables:

global MatPent,MatPentEdge,Map2a,Map2b,r0


#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#
#Load data:

p10=np.load('tmp/p10.npy'); p20=np.load('tmp/p20.npy'); p30=np.load('tmp/p30.npy'); p40=np.load('tmp/p40.npy')
StpDisp=np.load('tmp/StpDisp.npy'); PxMmR=np.load('tmp/PxMmR.npy');  r0=np.load('tmp/r0.npy'); r1=np.load('tmp/r1.npy'); r2=np.load('tmp/r2.npy')
MatPent=np.load('tmp/MatPent.npy'); MatPentEdge=np.load('tmp/MatPentEdge.npy'); MatPentS=np.load('tmp/MatPentS.npy'); MatPentL=np.load('tmp/MatPentL.npy')
begin=np.load('begin.npy'); stop=np.load('stop.npy')
thUv=-15;
#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#
#Define usefull functions:

##To test if a point is in a polygon:
def InsidePolygon(x,y,points):
    '''Return True if a coordinate (x, y) is inside a polygon defined by a list of verticies [(x1, y1), (x2, x2), ... , (xN, yN)].'''
    n=len(points)
    inside=False
    p1x,p1y=points[0]
    for i in range(1,n+1):
        p2x,p2y=points[i % n]
        if y>min(p1y, p2y):
            if y<=max(p1y, p2y):
                if x<=max(p1x, p2x):
                    if p1y!=p2y:
                        xinters=(y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if (p1x==p2x) or (x<=xinters):
                        inside=not inside
        p1x,p1y=p2x,p2y
    return inside

##To measure the overlap between binarized image and particle:
def OverlapVal(x):
    ##Define global variables:
    global MatPent,MatPentEdge,Map2a,Map2b,r0
    ##Extract data:
    Ia=x[0]; Ib=x[1]; A0=x[2];
    if (abs(Ia)<int(0.1*r0)) and (abs(Ib)<int(0.1*r0)): 
        try:
            ##Rotate pattern matrices:
            MatPentRot=(nd.interpolation.rotate(MatPent,A0,reshape=False,cval=0.0)>0.2).astype(float)
            MatPentEdgeRot=(nd.interpolation.rotate(MatPentEdge,A0,reshape=False,cval=0.0)>0.2).astype(float)
            ##Compute overlap value:
            Val0a=(np.sum(MatPentRot*Map2a[Map2a.shape[0]/2+Ia-MatPentRot.shape[0]/2:Map2a.shape[0]/2+Ia+MatPentRot.shape[0]/2,Map2a.shape[1]/2+Ib-MatPentRot.shape[1]/2:Map2a.shape[1]/2+Ib+MatPentRot.shape[1]/2])/np.sum(MatPentRot))
            Val0b=(np.sum(MatPentEdgeRot*Map2b[Map2b.shape[0]/2+Ia-MatPentEdgeRot.shape[0]/2:Map2b.shape[0]/2+Ia+MatPentEdgeRot.shape[0]/2,Map2b.shape[1]/2+Ib-MatPentEdgeRot.shape[1]/2:Map2b.shape[1]/2+Ib+MatPentEdgeRot.shape[1]/2])/np.sum(MatPentEdgeRot))
            Val0a=np.amax(np.array([0.1,Val0a]))
            Val0b=np.amax(np.array([0.1,Val0b]))
            if Val0a<0.5:
                Val0= 1 - 0.1 * Val0a - 0.9 * Val0b
            else:
                Val0=1 - 0.6*Val0a - 0.4*Val0b
        except:
            Val0=1
    else:
        Val0=1 + np.amax([abs(Ia)-int(0.1*r0),abs(Ib)-int(0.1*r0)])
    return Val0


#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#
#Etraction loop:

for itQ in range(begin,stop+1):
    if (itQ%2 == 1):
	continue
    print str(itQ)
    
    ##Define global variables:
    global MatPent,MatPentEdge,Map2a,Map2b,r0 
    
    ##Load the white and UV light images:
    ###Load:
    ImgWh=misc.imread('../pictures/'+'%04d'%itQ+'_Wh.jpg')
    ImgUv=misc.imread('../pictures/'+'%04d'%itQ+'_Uv.jpg')
    ###Extract the red and blue parts respectively:
    ImgWh=ImgWh[:,:,0]
    ImgUv=ImgUv[:,:,2]
    
    ###Reduce to the region of interest: 
    p1=np.array(p10); p2=np.array(p20)
    p3=np.array(p30); p4=np.array(p40)
    ###Make a mask:
    ImgMask=np.ones(ImgWh.shape)
    for it0 in range(0,ImgMask.shape[0]):
        for it1 in range(0,ImgMask.shape[1]):
            if ((it0<300) or (ImgMask.shape[0]-300<it0) or (it1<300) or (ImgMask.shape[1]-300<it1)):
                if (not InsidePolygon(it1,it0,[p1,p2,p3,p4])):
                    ImgMask[it0,it1]=0
    
    ##Reduce the images applying mask:
    ImgUv=ImgUv*ImgMask
    
    #-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#
    #Rough detection from UV points:
    
    ##Binarize the image:
    ###Blur picture:
    ImgUvBin=cv2.GaussianBlur(ImgUv.astype('uint8'),(7,7),0)
    ###Make adaptative threshold:
    ImgUvBin=cv2.adaptiveThreshold(ImgUvBin,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,141,thUv)
    
    ##Detect particle centers: 
    ###Measure picture islands:
    ImgUvLabel=nd.measurements.label(ImgUvBin)[0]
    ###Measurement centers of mass: 
    centers=nd.measurements.center_of_mass(ImgUvBin,ImgUvLabel,range(1,np.amax(ImgUvLabel)+1))
    areas=nd.measurements.sum(ImgUvBin,ImgUvLabel,range(1,np.amax(ImgUvLabel)+1))
    ###Remove fakes:
    centersN=[]
    for it0 in range(0,len(centers)):
        if (areas[it0]>40000)&(centers[it0][0] + 1.1*r0 < ImgWh.shape[0])&(centers[it0][1] + 1.1*r0 < ImgWh.shape[1])&(centers[it0][0] - 1.1*r0 > 0)&(centers[it0][1] - 1.1*r0 > 0):
            centersN.append(centers[it0])
    
    centers=centersN
    print(len(centers))
    #-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#
    #Accurate detection of the position and orientation:
    
    ##Loop over particles to optimize positionning and find orientation:
    VecI1=np.zeros(len(centers)); VecI2=np.zeros(len(centers)); VecA=np.zeros(len(centers));  VecVal=np.zeros(len(centers)); 
    for itPt in range(0,len(centers)):
        ###Extract rough position:
        I1=int(centers[itPt][0]); I2=int(centers[itPt][1]);
        ###Extract interesting sub-part of the matrix:
        Map0=ImgWh[I1-int(1.1*r0):I1+int(1.1*r0),I2-int(1.1*r0):I2+int(1.1*r0)]
        ###Threshold the the sub-matrix: 
        Map1=cv2.GaussianBlur(Map0.astype('uint8'),(int(Map0.shape[0]*0.02/2)*2+1,int(Map0.shape[0]*0.02/2)*2+1),0)
        ret,Map2a=cv2.threshold(Map1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        Map2a=(Map2a<100).astype(int)
        Map2b=cv2.adaptiveThreshold(Map1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        Map2b=(Map2b<100).astype(int)
        ###Optimization:
        count=3
        Val1=1
        while (Val1>0.2) and (count>0):
            x0=[0,0,count*20.]
            optim=optimize.minimize(OverlapVal,x0,method='Powell',options={'disp':False,'maxfev':150,'ftol':10**(-5),'xtol':10**(-5)}) 
            Val1=optim.fun
            if count==3:
                StoreDat=np.array([optim.x[0],optim.x[1],optim.x[2],Val1])
            else:
                StoreDat=np.vstack((StoreDat,np.array([optim.x[0],optim.x[1],optim.x[2],Val1])))
            count=count-1
        ###Store data:
        if count==2:
            VecI1[itPt]=I1+StoreDat[0]; VecI2[itPt]=I2+StoreDat[1]; VecA[itPt]=StoreDat[2]; VecVal[itPt]=Val1
        else:
            ValTot=StoreDat[:,3]
            I=np.where(ValTot==np.amin(ValTot))[0][0]
            VecI1[itPt]=I1+StoreDat[I,0]; VecI2[itPt]=I2+StoreDat[I,1]; VecA[itPt]=StoreDat[I,2]; VecVal[itPt]=StoreDat[I,3]
    #-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#
    #Graphical output:
    
    ##Position on image:
    ImgWh0=misc.imread('../pictures/'+'%04d'%itQ+'_Wh.jpg')
    plt.imshow(ImgWh0)
    for itPt in range(0,len(VecI1)):
        I1=VecI1[itPt]; I2=VecI2[itPt]; A0=-VecA[itPt]/180*m.pi;
        plt.plot([I2+r0*m.cos(A0+0*2*m.pi/5),I2+r0*m.cos(A0+1*2*m.pi/5),I2+r0*m.cos(A0+2*2*m.pi/5),I2+r0*m.cos(A0+3*2*m.pi/5),I2+r0*m.cos(A0+4*2*m.pi/5),I2+r0*m.cos(A0+5*2*m.pi/5)],[I1+r0*m.sin(A0+0*2*m.pi/5),I1+r0*m.sin(A0+1*2*m.pi/5),I1+r0*m.sin(A0+2*2*m.pi/5),I1+r0*m.sin(A0+3*2*m.pi/5),I1+r0*m.sin(A0+4*2*m.pi/5),I1+r0*m.sin(A0+5*2*m.pi/5)],linewidth=0.2,color='b')
    
    plt.xlim([0,ImgWh0.shape[1]])
    plt.ylim([0,ImgWh0.shape[0]])
    plt.axis('equal')
    plt.axis('off')
    plt.title('step '+'%04d'%itQ)
    plt.savefig('../movies/tmpPositionOnImage/'+'%04d'%itQ+'.png',dpi=250)
    plt.close()
    
    ##Position:
    for itPt in range(0,len(VecI1)):
        I1=VecI1[itPt]; I2=VecI2[itPt]; A0=-VecA[itPt]/180*m.pi;
        plt.plot([I2+r0*m.cos(A0+0*2*m.pi/5),I2+r0*m.cos(A0+1*2*m.pi/5),I2+r0*m.cos(A0+2*2*m.pi/5),I2+r0*m.cos(A0+3*2*m.pi/5),I2+r0*m.cos(A0+4*2*m.pi/5),I2+r0*m.cos(A0+5*2*m.pi/5)],[I1+r0*m.sin(A0+0*2*m.pi/5),I1+r0*m.sin(A0+1*2*m.pi/5),I1+r0*m.sin(A0+2*2*m.pi/5),I1+r0*m.sin(A0+3*2*m.pi/5),I1+r0*m.sin(A0+4*2*m.pi/5),I1+r0*m.sin(A0+5*2*m.pi/5)],linewidth=0.8,color='b')
    
    plt.xlim([0,ImgWh0.shape[1]])
    plt.ylim([0,ImgWh0.shape[0]])
    plt.axis('equal')
    plt.axis('off')
    plt.title('step '+'%04d'%itQ)
    plt.savefig('../movies/tmpPosition/'+'%04d'%itQ+'.png',dpi=250)
    plt.close()
    
   
    #-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#
    #Save data:
    
    ##Save position and pressure (|matrix coordinate 1| matrix coordinate 2 | angle (degree) | goodness of detection ([0,1])|):
    data=np.transpose(np.array([VecI1,VecI2,VecA,VecVal]))
    np.savetxt('../ParticleData/data_'+'%04d'%itQ,data,delimiter=' ',newline='\n')
















