#This code rename pictures, extract position and orientation of the pentagons, compute G2 for each pentagon and detect contacts for biax one way compression experiments:


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
from multiprocessing import Pool


#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#
#Input arguments:

##UV light image threshold:
thUv=-18

##Image number:
###First:
NImgF=1
###Last:
NImgL=2401

##Pixel to mm ratio:
PxMmR=7.81

##Displacement by step (mm)
StpDisp=0.5

##Number of parallel processing:
Npara=10

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


#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#
#Rename pictures:

for it0 in range(0,(NImgL-NImgF)/3+1):
    os.system('mv pictures/IMG_'+'%04d'%(NImgF+3*it0)+'.JPG pictures/'+'%04d'%(it0+1)+'_Uv.jpg')
    os.system('mv pictures/IMG_'+'%04d'%(NImgF+3*it0+1)+'.JPG pictures/'+'%04d'%(it0+1)+'_Wh.jpg')
    os.system('mv pictures/IMG_'+'%04d'%(NImgF+3*it0+2)+'.JPG pictures/'+'%04d'%(it0+1)+'_Pl.jpg')

#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#
#Crop pictures:

##Open the first white picture:
ImgWh=misc.imread('pictures/'+'%04d'%1+'_Wh.jpg');
##Select extrenum
plt.imshow(ImgWh)
plt.title('select 2 horizontal and vertical extremum points, then middle clic and close')
p1,p2,p3,p4=ginput(0,0)
plt.show()
plt.close()
##Crop all the pictures directly into the file:
###Extract limit coordinates:
I1m=int(p3[1]); I1M=int(p4[1]); I2m=int(p1[0]); I2M=int(p2[0]) 
###Define function to crop:
def Resize(it):
    global I1m,I1M,I2m,I2M
    os.system('convert pictures/'+'%04d'%it+'_Wh.jpg -crop '+str(I2M-I2m)+'x'+str(I1M-I1m)+'+'+str(I2m)+'+'+str(I1m)+' pictures/'+'%04d'%it+'_Wh.jpg > /dev/null')
    os.system('convert pictures/'+'%04d'%it+'_Pl.jpg -crop '+str(I2M-I2m)+'x'+str(I1M-I1m)+'+'+str(I2m)+'+'+str(I1m)+' pictures/'+'%04d'%it+'_Pl.jpg > /dev/null')
    os.system('convert pictures/'+'%04d'%it+'_Uv.jpg -crop '+str(I2M-I2m)+'x'+str(I1M-I1m)+'+'+str(I2m)+'+'+str(I1m)+' pictures/'+'%04d'%it+'_Uv.jpg > /dev/null')

###Paralelized resize of the pictures:
p=Pool(8)
p.map(Resize,range(1,(NImgL-NImgF)/3+2))

#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#
#Make evolution movies:
os.system('mkdir movies')
os.system('ffmpeg -r 10 -f image2 -i pictures/%04d_Wh.jpg -qscale 1 -vf scale=1500:1000 movies/MovieWh.avi >/dev/null 2>&1')
os.system('ffmpeg -r 10 -f image2 -i pictures/%04d_Pl.jpg -qscale 1 -vf scale=1500:1000 movies/MoviePl.avi >/dev/null 2>&1')
os.system('ffmpeg -r 10 -f image2 -i pictures/%04d_Uv.jpg -qscale 1 -vf scale=1500:1000 movies/MovieUv.avi >/dev/null 2>&1')


#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#
#Initialization:

##Load the first white images:
ImgWh=misc.imread('pictures/0001_Wh.jpg')

##Select the initial area of interest:
plt.imshow(ImgWh)
plt.colorbar()
plt.title('select 4 corners trigonometrically from upper left, then middle clic and close')
p10,p20,p30,p40=ginput(0,0)
plt.show()
plt.close()

##Make the pentagon patterns:
###Select edges:
plt.imshow(ImgWh)
plt.title('zoom and clic on 5 edges of a pentagon, then middle clic and close')
_,q1,q2,q3,q4,q5=ginput(0,0)
plt.show()
plt.close()
###Compute pentagone radius:
r0=int((0.5/m.sin(m.pi/5))*np.mean(np.array([m.sqrt((q1[0]-q2[0])**2+(q1[1]-q2[1])**2),m.sqrt((q2[0]-q3[0])**2+(q2[1]-q3[1])**2),m.sqrt((q3[0]-q4[0])**2+(q3[1]-q4[1])**2),m.sqrt((q4[0]-q5[0])**2+(q4[1]-q5[1])**2),m.sqrt((q5[0]-q1[0])**2+(q5[1]-q1[1])**2)])))
###Build the matrix:
MatPent=np.zeros([2*int(r0*1.),2*int(r0*1.)])
q1=r0+np.array([r0*m.cos(0*2*m.pi/5),r0*m.sin(0*2*m.pi/5)]); q2=r0+np.array([r0*m.cos(1*2*m.pi/5),r0*m.sin(1*2*m.pi/5)]); q3=r0+np.array([r0*m.cos(2*2*m.pi/5),r0*m.sin(2*2*m.pi/5)]); q4=r0+np.array([r0*m.cos(3*2*m.pi/5),r0*m.sin(3*2*m.pi/5)]); q5=r0+np.array([r0*m.cos(4*2*m.pi/5),r0*m.sin(4*2*m.pi/5)])  
for it0 in range(0,MatPent.shape[0]):
    for it1 in range(0,MatPent.shape[1]):
        if InsidePolygon(it1+1,it0+1,[q1,q2,q3,q4,q5]):
            MatPent[it0,it1]=1

###Make the pattern of edges:
MatPentEdge=cv2.Canny(1-MatPent.astype('uint8'),0,1)
kernel=np.ones((int(MatPentEdge.shape[0]*0.005)*2+1,int(MatPentEdge.shape[0]*0.005)*2+1),np.uint8)
MatPentEdge=cv2.dilate(MatPentEdge,kernel,iterations=1)
###Make a smaller pentagon pattern:
r1=int(0.97*r0)
MatPentS=np.zeros([2*int(r1*1.),2*int(r1*1.)])
q1=r1+np.array([r1*m.cos(0*2*m.pi/5),r1*m.sin(0*2*m.pi/5)]); q2=r1+np.array([r1*m.cos(1*2*m.pi/5),r1*m.sin(1*2*m.pi/5)]); q3=r1+np.array([r1*m.cos(2*2*m.pi/5),r1*m.sin(2*2*m.pi/5)]); q4=r1+np.array([r1*m.cos(3*2*m.pi/5),r1*m.sin(3*2*m.pi/5)]); q5=r1+np.array([r1*m.cos(4*2*m.pi/5),r1*m.sin(4*2*m.pi/5)])  
for it0 in range(0,MatPentS.shape[0]):
    for it1 in range(0,MatPentS.shape[1]):
        if InsidePolygon(it1+1,it0+1,[q1,q2,q3,q4,q5]):
            MatPentS[it0,it1]=1

###Make a Larger pentagon pattern:
r2=int(1.1*r0)
MatPentL=np.zeros([2*int(r2*1.),2*int(r2*1.)])
q1=r2+np.array([r2*m.cos(0*2*m.pi/5),r2*m.sin(0*2*m.pi/5)]); q2=r2+np.array([r2*m.cos(1*2*m.pi/5),r2*m.sin(1*2*m.pi/5)]); q3=r2+np.array([r2*m.cos(2*2*m.pi/5),r2*m.sin(2*2*m.pi/5)]); q4=r2+np.array([r2*m.cos(3*2*m.pi/5),r2*m.sin(3*2*m.pi/5)]); q5=r2+np.array([r2*m.cos(4*2*m.pi/5),r2*m.sin(4*2*m.pi/5)])  
for it0 in range(0,MatPentL.shape[0]):
    for it1 in range(0,MatPentL.shape[1]):
        if InsidePolygon(it1+1,it0+1,[q1,q2,q3,q4,q5]):
            MatPentL[it0,it1]=1

##Make folders to save data:
os.system('mkdir movies/tmpPositionOnImage')
os.system('mkdir movies/tmpPosition')
os.system('mkdir movies/tmpPositionContact')
os.system('mkdir ParticleData')

##Save global variables
os.system('mkdir tmp')
np.save('tmp/p10.npy',p10); np.save('tmp/p20.npy',p20); np.save('tmp/p30.npy',p30); np.save('tmp/p40.npy',p40)
np.save('tmp/StpDisp.npy',StpDisp); np.save('tmp/PxMmR.npy',PxMmR); np.save('tmp/thUv.npy',thUv); np.save('tmp/r0.npy',r0); np.save('tmp/r1.npy',r1); np.save('tmp/r2.npy',r2)
np.save('tmp/MatPent.npy',MatPent); np.save('tmp/MatPentEdge.npy',MatPentEdge); np.save('tmp/MatPentS.npy',MatPentS); np.save('tmp/MatPentL.npy',MatPentL)

#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#-|\/|-#
#Prepare manual parallelization:

##Segmentation vector:
Segment=np.linspace(1,(NImgL-NImgF)/3+2,Npara+1).astype(int)

##Segmentation loop:
for itCp in range(1,Npara+1):
    ###Create computation folder:
    os.system('mkdir Compute_'+'%02d'%itCp)
    ###Transfer global data:
    os.system('cp -r tmp Compute_'+'%02d'%itCp+'/tmp')
    ###Store begining and end:
    np.save('Compute_'+'%02d'%itCp+'/begin.npy',Segment[itCp-1])
    if itCp<Npara:
        np.save('Compute_'+'%02d'%itCp+'/stop.npy',Segment[itCp]-1)
    else:
        np.save('Compute_'+'%02d'%itCp+'/stop.npy',Segment[itCp])
    ###Transfer main code:
    os.system('cp Main.py Compute_'+'%02d'%itCp+'/Main.py')


print 'Run each code'



