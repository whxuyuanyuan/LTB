#Load usefull libraries:
import numpy as np
import math as m
from scipy import misc
from scipy import optimize
from scipy import stats
from scipy import ndimage as nd
import cv2
import os

#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#
#Define global variables:

global MatPent,MatPentEdge,Map2a,Map2b,r0


#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#
#Load data:

p10=np.load('tmp/p10.npy'); p20=np.load('tmp/p20.npy'); p30=np.load('tmp/p30.npy'); p40=np.load('tmp/p40.npy')
StpDisp=np.load('tmp/StpDisp.npy'); PxMmR=np.load('tmp/PxMmR.npy'); r0=np.load('tmp/r0.npy'); r1=np.load('tmp/r1.npy'); r2=np.load('tmp/r2.npy')
MatPent=np.load('tmp/MatPent.npy'); MatPentEdge=np.load('tmp/MatPentEdge.npy'); MatPentS=np.load('tmp/MatPentS.npy'); MatPentL=np.load('tmp/MatPentL.npy')
begin=np.load('begin.npy'); stop=np.load('stop.npy')

thUv=-18;
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
    
    print str(itQ)
    ImgWh=misc.imread('../pictures/'+'%04d'%itQ+'_Wh.jpg')
    ###Reduce to the region of interest: 
    p1=np.array(p10); p2=np.array(p20)
    p3=np.array(p30); p4=np.array(p40)

    # Import data
    fileInput = open('../ParticleData/data_'+'%04d'%itQ)
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
    VecA = np.array(VecA)
    #-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#
    #Measurement of the pressure:
    
    ##Load and reduce the polarized picture:
    ImgPl=misc.imread('../pictures/'+'%04d'%itQ+'_Pl.jpg')
    ImgPl=ImgPl[:,:,1]
    ImgPl1 = ImgPl.astype('float')
    
    ##Computation of G^2 for the full map:
    ###Compute G2 for each pixel:
    ImG2=np.zeros(ImgPl.shape)
    for it0 in range(0,ImG2.shape[0]):
        for it1 in range(0,ImG2.shape[1]):
            if (it0>0) and (it1>0) and (it0<ImgPl1.shape[0]-1) and (it1<ImgPl1.shape[1]-1): 
                ImG2[it0,it1]=1/4.*((ImgPl1[it0+1,it1]-ImgPl1[it0-1,it1])**2.+(ImgPl1[it0,it1+1]-ImgPl1[it0,it1-1])**2.+1/2.*(ImgPl1[it0+1,it1+1]-ImgPl1[it0-1,it1-1])**2.+1/2.*(ImgPl1[it0+1,it1-1]-ImgPl1[it0-1,it1+1])**2.)
    
    ##Measurement of G2 for each particle:
    ###Loop over particles:
    VecG2=np.zeros(len(VecI1));
    for itPt in range(0,len(VecI1)):
        ####Load properties of current particle:
        I1c=VecI1[itPt]; I2c=VecI2[itPt]; Ac=VecA[itPt]
        ####Rotate pattern:
        MatPentSRot=(nd.interpolation.rotate(MatPentS,Ac,reshape=False,cval=0.0)>0.2).astype(float)
        ####Compute integral:
        VecG2[itPt]=np.sum(ImG2[int(I1c-MatPentS.shape[0]/2):int(I1c-MatPentS.shape[0]/2)+MatPentS.shape[0], int(I2c-MatPentS.shape[1]/2):int(I2c-MatPentS.shape[1]/2)+MatPentS.shape[1]]*MatPentSRot)
  


    
    #-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#
    #Measurement of the contacts:
    
    ##Loop over particles:
    VecCont=np.array([0,0,0,0])
    ImgPl2=stats.threshold(ImgPl1,threshmin=40,threshmax=200,newval=0)
    for itPt in range(0,len(VecI1)):
        ###Load properties of current particle:
        I1c=VecI1[itPt]; I2c=VecI2[itPt]; Ac=VecA[itPt]
        ###Detect neighbors:
        d0=np.sqrt((I1c-VecI1)**2+(I2c-VecI2)**2)
        In=np.where((d0<2*r2) & (d0>0))[0]
        if len(In)>0:
            ###Rotate main patterns:
            MatPentLRot0=(nd.interpolation.rotate(MatPentL,Ac,reshape=False,cval=0.0)>0.2).astype(float)
            MatPentRot0=(nd.interpolation.rotate(MatPent,Ac,reshape=False,cval=0.0)>0.2).astype(float)
            ###Loop over neighbors:
            for itPtb in range(0,len(In)):
                ###Load properties of neighbors particle:
                I1d=VecI1[In[itPtb]]; I2d=VecI2[In[itPtb]]; Ad=VecA[In[itPtb]]
                ###Rotate the neighbor patterns:
                MatPentLRot1=(nd.interpolation.rotate(MatPentL,Ad,reshape=False,cval=0.0)>0.2).astype(float)
                MatPentRot1=(nd.interpolation.rotate(MatPent,Ad,reshape=False,cval=0.0)>0.2).astype(float)
                ###Make the overlapping map:
                ImgCont=np.zeros(ImgPl2.shape);
		try:
                	ImgCont[int(I1c-MatPentL.shape[0]/2):int(I1c-MatPentL.shape[0]/2)+MatPentL.shape[0],int(I2c-MatPentL.shape[1]/2):int(I2c-MatPentL.shape[1]/2)+MatPentL.shape[1]]=ImgCont[int(I1c-MatPentL.shape[0]/2):int(I1c-MatPentL.shape[0]/2)+MatPentL.shape[0],int(I2c-MatPentL.shape[1]/2):int(I2c-MatPentL.shape[1]/2)+MatPentL.shape[1]]+MatPentLRot0
		except:
			temp = ImgCont[int(I1c-MatPentL.shape[0]/2):int(I1c-MatPentL.shape[0]/2)+MatPentL.shape[0],int(I2c-MatPentL.shape[1]/2):int(I2c-MatPentL.shape[1]/2)+MatPentL.shape[1]]
			ImgCont[int(I1c-MatPentL.shape[0]/2):int(I1c-MatPentL.shape[0]/2)+MatPentL.shape[0],int(I2c-MatPentL.shape[1]/2):int(I2c-MatPentL.shape[1]/2)+MatPentL.shape[1]]=ImgCont[int(I1c-MatPentL.shape[0]/2):int(I1c-MatPentL.shape[0]/2)+MatPentL.shape[0],int(I2c-MatPentL.shape[1]/2):int(I2c-MatPentL.shape[1]/2)+MatPentL.shape[1]]+MatPentLRot0[0:temp.shape[0], 0:temp.shape[1]]
		try:
                	ImgCont[int(I1d-MatPentL.shape[0]/2):int(I1d-MatPentL.shape[0]/2)+MatPentL.shape[0],int(I2d-MatPentL.shape[1]/2):int(I2d-MatPentL.shape[1]/2)+MatPentL.shape[1]]=ImgCont[int(I1d-MatPentL.shape[0]/2):int(I1d-MatPentL.shape[0]/2)+MatPentL.shape[0],int(I2d-MatPentL.shape[1]/2):int(I2d-MatPentL.shape[1]/2)+MatPentL.shape[1]]+MatPentLRot1
		except:
			temp = ImgCont[int(I1d-MatPentL.shape[0]/2):int(I1d-MatPentL.shape[0]/2)+MatPentL.shape[0],int(I2d-MatPentL.shape[1]/2):int(I2d-MatPentL.shape[1]/2)+MatPentL.shape[1]]
			ImgCont[int(I1d-MatPentL.shape[0]/2):int(I1d-MatPentL.shape[0]/2)+MatPentL.shape[0],int(I2d-MatPentL.shape[1]/2):int(I2d-MatPentL.shape[1]/2)+MatPentL.shape[1]]=ImgCont[int(I1d-MatPentL.shape[0]/2):int(I1d-MatPentL.shape[0]/2)+MatPentL.shape[0],int(I2d-MatPentL.shape[1]/2):int(I2d-MatPentL.shape[1]/2)+MatPentL.shape[1]]+MatPentLRot1[0:temp.shape[0], 0:temp.shape[1]]
                ###If there i overlapping:
                if np.amax(ImgCont)>1:
                    ###Expend the overlap:
                    ImgCont=(ImgCont>1).astype('uint8')
                    kernel=np.ones((11,11),np.uint8)
                    ImgCont1=cv2.dilate(ImgCont,kernel,iterations=1)
                    ###Measure polarization intensity at the overlap:
                    ImgTmp=np.zeros(ImgPl2.shape); ImgTmp[int(I1c-MatPent.shape[0]/2):int(I1c-MatPent.shape[0]/2)+MatPent.shape[0],int(I2c-MatPent.shape[1]/2):int(I2c-MatPent.shape[1]/2)+MatPent.shape[1]]=ImgTmp[int(I1c-MatPent.shape[0]/2):int(I1c-MatPent.shape[0]/2)+MatPent.shape[0],int(I2c-MatPent.shape[1]/2):int(I2c-MatPent.shape[1]/2)+MatPent.shape[1]]+MatPentRot0
                    Val00=np.sum(ImgCont1*ImgTmp*ImgPl2)
                    ImgTmp=np.zeros(ImgPl2.shape); ImgTmp[int(I1c-MatPent.shape[0]/2):int(I1c-MatPent.shape[0]/2)+MatPent.shape[0],int(I2c-MatPent.shape[1]/2):int(I2c-MatPent.shape[1]/2)+MatPent.shape[1]]=ImgTmp[int(I1c-MatPent.shape[0]/2):int(I1c-MatPent.shape[0]/2)+MatPent.shape[0],int(I2c-MatPent.shape[1]/2):int(I2c-MatPent.shape[1]/2)+MatPent.shape[1]]+MatPentRot1
                    Val01=np.sum(ImgCont1*ImgTmp*ImgPl2)
                    if (Val00>1800) and (Val01>1800):
                        ImgTmp=ImgCont1*ImgPl2
                        I1cont,I2cont=np.where(ImgCont>0); I1cont=np.mean(I1cont); I2cont=np.mean(I2cont); 
                        VecCont=np.vstack((VecCont,np.array([itPt,In[itPtb],I1cont,I2cont])))
    
    VecCont=np.delete(VecCont,0,0)
    


    ##Save contacts (|number particle 1 | number particle 2 | contact matrix coordinate 1| contact matrix coordinate 2) :
    np.savetxt('../ParticleData/contact_'+'%04d'%itQ,VecCont,delimiter=' ',newline='\n')

















