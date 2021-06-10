# -*- coding: utf-8 -*-
"""
Created on Mon May 10 13:39:59 2021

@author: phili
"""


from scipy.spatial.distance import directed_hausdorff

from matplotlib import animation
from scipy      import optimize
from mpl_toolkits.mplot3d import Axes3D 


import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import cv2

from scipy.spatial.distance import directed_hausdorff

from shapely.geometry import Point, LineString, Polygon ,MultiPolygon
from descartes import PolygonPatch
import shapely.ops as so

import pandas as pd
import xlsxwriter

#%%
##
##
##
##
##
##
#%%


#Coordenadas = GInterface.mainWindow()


#%%
class Method:
    
    def __init__(self,Data):
        
        self.Data=Data
        
        dropArea    = Data[0]
        needleArea  = Data[1]
        imag        = Data[2]
        
        Dens1       =Data[3]
        Dens2       =Data[4]
        Thick       =Data[5]
        
        Density1 = Dens1
        Density2 = Dens2   

        Density = Density1-Density2 
        needleDiameter = Thick*1e-3
        g = 9.81          # Gravedad

          
        image = imag
        GrayImage=  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("-------------------------------------------------------------")
        print("Image Properties:")
        print("")
        print("Image size")
        print("Xsize:",GrayImage[0,:].size)
        print("Ysize:",GrayImage[:,0].size)
        
        print("--------------------------------------------------------------")
        
        print("Density Drop",Density1)
        print("Density Medium",Density2)
        print("Needle Thickness:",Thick)
        print("--------------------------------------------------------------")
        
        
        def showing(img,string):
     
            plt.title(string)
            plt.imshow(img)
            plt.show()


        
        def findImageSize(img):
    
            imXsize = img[0,:].size
            imYsize = img[:,0].size
            
            imxCenter= int(imXsize/2)
            imyCenter= int(imYsize/2)
        #    
            return imXsize, imYsize
        
        size = findImageSize(GrayImage)

        
        
        
        def filterImage(img):
            edges = cv2.Canny(img,20,200)
            return edges
        
        edges =filterImage(GrayImage)
        
        showing(GrayImage,"Gray Image")
        showing(edges,"Edge detection Image")
        
        
        def ShowInit(img,dropArea,needleArea):
            
            print("drop")
            print(dropArea)
            
            dix=dropArea[0,0]
            diy=dropArea[0,1]
            dfx=dropArea[0,2]
            dfy=dropArea[0,3]
            
            
            nix=needleArea[0,0]
            niy=needleArea[0,1]
            nfx=needleArea[0,2]
            nfy=needleArea[0,3]
            
            radioN2= (nfx-nix)+40
        #    radioN = 350
            
            cx=dix+((dfx-dix)/2)
            cy=diy+((dfy-diy)/2)
            
            print("radio de inicialización",radioN2)
            s = np.linspace(0, 2*np.pi, 400)
            r =[cy]  + (radioN2*np.sin(s)/1.0)#  500
            c =[cx]  + (radioN2*np.cos(s)/1.2) #1
            init = np.array([r, c]).T
            
            
            plt.figure()
            plt.title(" Inicialización Para segmentar con Snakes")
            cv2.rectangle(img,(dix,diy),(dfx,dfy),(0,255,0),3)
            cv2.rectangle(img,(nix,niy),(nfx,nfy),(0,255,0),3)
            plt.imshow(img,cmap="gray")
            plt.plot(init[:,1],init[:,0],color="red")
            plt.scatter(cx,cy,color="red")
            plt.show()
            
            return
        #
        ShowInit(image,dropArea,needleArea)
        
        print("Waiting for Snake Segmentation")
        
        def useSnake(img,alphaValue):
            print("useSnake function")
             
            dix=dropArea[0,0]
            diy=dropArea[0,1]
            dfx=dropArea[0,2]
            dfy=dropArea[0,3]
            
            
            nix=needleArea[0,0]
            niy=needleArea[0,1]
            nfx=needleArea[0,2]
            nfy=needleArea[0,3]
            
            cx=dix+((dfx-dix)/2)
            cy=diy+((dfy-diy)/2)
            
            radioN2= (nfx-nix)+10
        #    radioN =430
        
            s = np.linspace(0, 2*np.pi, 400)
            r =cy+ (radioN2*np.sin(s)/1.0)#  500
            c =cx + (radioN2*np.cos(s)/1.3) #1000 for ddd7
            init = np.array([r, c]).T
            
            snake = active_contour(gaussian(img, 3),init, alpha=alphaValue, beta=10, gamma=0.001,coordinates='rc') 
            
            
            fig = plt.figure()
            ax=plt.subplot()
            plt.title("Borde segmentado con Snake")
            plt.imshow(img,cmap="gray")
            plt.plot(snake[:,1],snake[:,0],color ="red",label="Contorno Activo")
            ax.legend(loc="best")
            plt.grid()
            plt.show()   
            
            return snake
        
        
#        GraySnake1 = useSnake(GrayImage,0.010)
        
        
        def HaussDorffDistance(img,snake1,snake2):
            
            HausDorffDistance=directed_hausdorff(snake1,snake2)    
            fig = plt.figure()
            ax=plt.subplot()
            plt.title("Comparar PointClouds y calcular HausDorfff")
            plt.imshow(img,cmap="gray")
            
            plt.plot(snake1[:,1],snake1[:,0],label="GrayImage")
        #    plt.plot(snake2[:,1],snake2[:,0],label="edgeImage")
            ax.legend(loc="best")
            plt.grid()
            plt.show()  
            
        
            
            
            return HausDorffDistance
            
        #HaussDiff =HaussDorffDistance(edges,GraySnake1,GraySnake7)
        
        def AreaShoeLace(Array):
            
            xArray=Array[:,0]
            yArray=Array[:,1]
            
            sumaSs1=0
            sumaSs2=0    
            
            for i in range(0,xArray.size-1,1):
                
                sumaSs1 = sumaSs1+ xArray[i]*yArray[i+1]
                sumaSs2 = sumaSs2+ xArray[i+1]*yArray[i]
                
            smSs1 = sumaSs1+ xArray[xArray.size-1]*yArray[0]
            smSs2 = sumaSs2+ xArray[0]*yArray[xArray.size-1]
            
            Area= (abs(smSs1-smSs2))/2
            return Area
        
        ##    
        #GrayArea = AreaShoeLace(GraySnake1)
        #edgeArea = AreaShoeLace(GraySnake7)
        #print("Areas de ambas segmentaciones : ",GrayArea,edgeArea)
        
        def poly(snake1,snake2):
            
            poly1 = Polygon(snake1)
            poly2 = Polygon(snake2)
            
            poly1_area = poly1.area
            poly2_area = poly2.area
            
            return poly1_area,poly2_area
        #
        #areas=poly(GraySnake1,GraySnake1)
        
            
        #print("Valores Areas con Poly :",areas)
        #print("Valores Areas con shoe : ",GrayArea,edgeArea)
        
        
        def DiceForPolyongs(Array1,Array2):
            
                
            BLUE = '#6699cc'
            GRAY = '#C7D322'
            CIAN = '#59C1DA'
        
            
            poly1 = Polygon(Array1)
            poly2 = Polygon(Array2)
            
            poly1_area = poly1.area
            poly2_area = poly2.area
            
            intersectionUNO  = poly1.intersection(poly2).area
            intersectionDOS  = poly1.intersection(poly1).area
            intersectionTRES = poly2.intersection(poly2).area
            
            Dice = 2*intersectionUNO/(intersectionDOS+intersectionTRES)
            
            new_shape= so.unary_union([poly1, poly2])
            
            
            fig = plt.figure() 
            plt.title(" Intersección entre poligonos usando shapely  Snake/Freenab")
            ax = fig.gca() 
            ax.add_patch(PolygonPatch(poly2, fc=CIAN, ec=BLUE, alpha=1, zorder=1 ))
            ax.add_patch(PolygonPatch(poly1, fc=GRAY, ec=BLUE, alpha=1, zorder=1 ))
            #ax.add_patch(PolygonPatch(poly2, fc=CIAN, ec=BLUE, alpha=1, zorder=1 ))
            ax.axis('equal')
            plt.grid()
            plt.show()
        
        
            return 100*Dice
        ##
        #Dice=DiceForPolyongs(GraySnake1,GraySnake7)
        #print("Dice Coeff:",Dice)
        
        
        #%% Segmentación de la gota entera para obtener ñps dapds
        #%%
            
        
        def getNeedle(size,img,needleArea,image):
            
            print("getNeedle function")
            print("This needle width")
            print(needleArea)
            
            nix=needleArea[0,0]
            niy=needleArea[0,1]
            nfx=needleArea[0,2]
            nfy=needleArea[0,3]
            
            needle2 = []
            
            for i in range(niy,nfy,1):
                
                for j in range((nfx-nix),size-30,1):
                    
                    if img[i,j] == 255:
                        
                        np.array([needle2.append([j,i])])
                        
                        needleArray = np.array(needle2) 
                        
            plt.figure()
            plt.title("Needle sides' Segmented")
            cv2.rectangle(img,(nix,niy),(nfx,nfy),(0,0,255),3)
            plt.imshow(image)
            plt.scatter(needleArray[:,0],needleArray[:,1])
            plt.show()   
            
            return needleArray
        
        
        
        NeedleArray=getNeedle(size[0],edges,needleArea,image)
        
        
        def NeedleWidth(needleArray):
            print("NeedleWidth Function")
            
            needleSize = needleArray[:,0].size
            diff = []
            
            for i in range(0,needleSize-2,2):
                
                np.array([diff.append([needleArray[i+1,0]-needleArray[i,0]])])
                
                diffArray= np.array(diff)
                absDiff = abs(diffArray)
                
                realDiff =[]
                
                for i in range(0,len(absDiff),1):
                    if absDiff[i]>2:
                        
                        realDiff.append(absDiff[i])
                        
            realDiff = np.array(realDiff)
            needleWidth = int(np.mean(realDiff))
            
            return needleWidth
        #
        #
        Nwidth = NeedleWidth(NeedleArray)
        print("Needle width:",Nwidth,"Pixel Units")
        
        
        
        #%%
        
        def Ratio(width):
            print("Ratio Function")
            
            ratio = needleDiameter/width
            
            return ratio
        
        ratio = Ratio(Nwidth)
        
        
        
        #%%
        
        
        def NeedleCenter(needleArray):
            print("NeedleCenter Function")
            
            minNeedle = np.min(needleArray[:,0])
            maxNeedle = np.max(needleArray[:,0])
            
            centerNeedle = int((minNeedle+maxNeedle)/2)
            
            plt.figure()
            plt.title("Center of needle")
            plt.imshow(edges, cmap="gray")
            plt.scatter(needleArray[:,0],needleArray[:,1])
            plt.scatter(centerNeedle,needleArray[0,1],color="red")
            plt.show()   
            
            return centerNeedle
        
        Ncenter = NeedleCenter(NeedleArray)
        
        
        
        
        #%%
        
        def getApex(imYsize,centerNeedle,needleArray,img,needleArea,dropArea):
            print("getApex Function:")
            
            
            dix=dropArea[0,0]  # drop inicial x
            diy=dropArea[0,1]  # drop inicial y
            dfx=dropArea[0,2]  # drop inicial x
            dfy=dropArea[0,3]  #
            
            
            nix=needleArea[0,0]
            niy=needleArea[0,1]
            nfx=needleArea[0,2]
            nfy=needleArea[0,3]
            
            
            apexPoint  = []
            
        #    for y in range(needleArray[0,1]+500,imYsize,1):
            for y in range(diy+100,dfy,1):    
                
                while img[y,centerNeedle]!=0:
                    
                    np.array([apexPoint.append([centerNeedle,y])])
                    
                    break
                
                
            apex = np.array(apexPoint)
        
            print("apex,",apex)
            
            
            apexRegion=[]
            
            for x in range(needleArray[0,0],needleArray[1,0],1):
                
                for y in range(apex[0,1]-3,apex[0,1]+3,1):        
                    if edges[y,x]==255:
                        
                        np.array([apexRegion.append([x,y])])
                        apexRegionArray= np.array(apexRegion)
                        
            plt.figure()
            plt.title("Apex region ")
            plt.imshow(edges, cmap="gray")
            plt.scatter(needleArray[:,0],needleArray[:,1])
            plt.scatter(centerNeedle,needleArray[0,1],color ="red")
            plt.scatter(apexRegionArray[:,0],apexRegionArray[:,1])
        #   
            
            
            apexApex = np.max(apexRegionArray[:,1])
            apexSize = apexRegionArray[:,1].size
            
            apexPosition=[]
            
            for x in range(0,apexSize,1):
                
                
                if apexRegionArray[x,1]==apexApex:
                    
                    np.array([apexPosition.append([x])])
                    apexPositionArray=np.array(apexPosition)
                    
            minPos=np.min(apexPositionArray)
            maxPos=np.max(apexPositionArray)
            
            apexLine = int((apexRegionArray[minPos,0]+apexRegionArray[maxPos,0])/2)
            apexFin = np.array([apexLine,apexApex])
            
            plt.figure()
            plt.title("Apex Position ")
            plt.imshow(img, cmap="gray")
            plt.scatter(apexFin[0] ,apexFin[1] )
        #    plt.plot(GraySnake1[:,1],GraySnake1[:,0])
            plt.show()
            
            print("apex position",apexFin)
            return apexFin
        
        
        
        apex = getApex(size[1],Ncenter,NeedleArray,edges,needleArea,dropArea)
        
        
        
        #%%
        
        def PlotRadio(snake1,snake2):
                
            plt.figure()
            plt.imshow(GrayImage,cmap="gray")
            plt.plot(snake1[:,1],snake1[:,0])
            plt.scatter(snake1[73:122,1],snake1[73:122,0],color ="red")
            plt.plot(snake2[65:135,1],snake2[65:135,0],color ="blue")
            plt.show()
                
        
        
        def getRadio1(img,snake):
            
            
            a=70
            b=129
            
            x =snake[a:b,0]
            y =snake[a:b,1]
            
            x_m = np.mean(x)
            y_m = np.mean(y)
            
            x_mArray = np.ones(x.size)*x_m
            y_mArray = np.ones(y.size)*x_m
            
            u2 = x - x_m
            v2 = y - y_m
            
            Suv2  = np.sum(u2*v2)
            Suu2  = np.sum(u2**2)
            Svv2  = np.sum(v2**2)
            Suuv2 = np.sum(u2**2 * v2)
            Suvv2 = np.sum(u2 * v2**2)
            Suuu2 = np.sum(u2**3)
            Svvv2 = np.sum(v2**3)
            
            A2 = np.array([ [ Suu2, Suv2 ], [Suv2, Svv2]])
            B2 = np.array([ Suuu2 + Suvv2, Svvv2 + Suuv2 ])/2.0
            uc2, vc2 = np.linalg.solve(A2, B2)
            xc_2 = x_m + uc2
            yc_2 = y_m + vc2
            Ri_2     = np.sqrt((x-xc_2)**2 + (y-yc_2)**2)
            R_2      = np.mean(Ri_2)
            
            residu_2 = np.sum((Ri_2-R_2)**2)
            
                
        
            fig = plt.figure()
            ax=plt.subplot()
            plt.title("contorno activo + radio")
            plt.scatter(GraySnake1[:,1],GraySnake1[:,0],color="red",label="Contorno Activo")
            plt.imshow(img,cmap="gray")
            plt.scatter(y,x)
            plt.scatter(yc_2,xc_2)
                
            ax.legend(loc="best")
            plt.show()
            
            
            
            return R_2
        
        
        
        
        def getRadio2(snake,apex,DropArea):
            
            dropArea=DropArea
            
            dix=dropArea[0,0]  # drop inicial x
            diy=dropArea[0,1]  # drop inicial y
            dfx=dropArea[0,2]  # drop final   x
            dfy=dropArea[0,3]  # drop final   y
            
            
            Distance = 68
            
            imYsize = snake[:,0].size
            
            Xinicio = apex[0]-Distance
            Xfin    = apex[0]+Distance
            
            surface=[]
            for i in range(Xinicio,Xfin,1):
                
                j =  dfy-2
                
#                j =  imYsize-10
                
                while j > 0:
                    
                    if snake[j,i] == 255:                                   # aca recogo datos de la superficie
                        
                        np.array([surface.append([i,j])])
                        
                        break
                    
                    j -= 1
                    
            surface
            surfaceArray=np.array(surface)
            
            
            
            
            plt.figure()
            plt.title("Apex surface")
            plt.imshow(snake,cmap="gray")
            plt.scatter(surfaceArray[:,0],surfaceArray[:,1])
            plt.show()
            
            x =surfaceArray[:,0]
            y =surfaceArray[:,1]
            
            
            x_m = np.mean(x)
            y_m = np.mean(y)
            
            x_mArray = np.ones(x.size)*x_m
            y_mArray = np.ones(y.size)*x_m
            
            
            u2 = x - x_m
            v2 = y - y_m
            
            Suv2  = np.sum(u2*v2)
            Suu2  = np.sum(u2**2)
            Svv2  = np.sum(v2**2)
            Suuv2 = np.sum(u2**2 * v2)
            Suvv2 = np.sum(u2 * v2**2)
            Suuu2 = np.sum(u2**3)
            Svvv2 = np.sum(v2**3)
            
            A2 = np.array([ [ Suu2, Suv2 ], [Suv2, Svv2]])
            
            B2 = np.array([ Suuu2 + Suvv2, Svvv2 + Suuv2 ])/2.0
            uc2, vc2 = np.linalg.solve(A2, B2)
            xc_2 = x_m + uc2
            yc_2 = y_m + vc2
            
            Ri_2     = np.sqrt((x-xc_2)**2 + (y-yc_2)**2)
            R_2      = np.mean(Ri_2)
            residu_2 = np.sum((Ri_2-R_2)**2)
            
            return R_2,xc_2,yc_2
          
        Rvalues =getRadio2(edges,apex,dropArea)
            
        print(Rvalues)
        
        
            
        
            
        #%%
        
        def getGamma(Radio,yc_2,imXsize,imYsize,apexFin,IMG,DropArea):
            
            dropArea = DropArea
            
            dix=dropArea[0,0]  # drop inicial x
            diy=dropArea[0,1]  # drop inicial y
            dfx=dropArea[0,2]  # drop final   x
            dfy=dropArea[0,3]  # drop final   y
            
            
            
            
            Yinicio = int(yc_2)-100
            Yfin    = int(yc_2)+100
            
            Lone   = np.linspace(1,imXsize,imXsize)
        
            lin  = Yinicio *np.ones((1,len(Lone)))
            lfin = Yfin*np.ones((1,len(Lone)))
        
            dropSide1 = []
            Xlimite = imXsize-30
            
            for i in range(Yinicio,Yfin,1):
                
                for j in range(150,Xlimite,1):
                    
                    if IMG[i,j] == 255:
                        
                        np.array([dropSide1.append([j,i])])
                        
                        break
           
            
            dropSide1
            dropSideArray1 = np.array(dropSide1) 
            
            dropSide2 = []
            Xlimite = imXsize-30
            
            for i in range(Yinicio,Yfin,1):
                
                for j in range(150,Xlimite,1):
                    
                    jj = Xlimite-j
                    
                    if IMG[i,Xlimite-j] == 255:
                        
                        np.array([dropSide2.append([jj,i])])
                        
                        break
                         
            dropSide2
            dropSideArray2 = np.array(dropSide2) 
            
            Left= np.min(dropSideArray1[:,0])
            Right=np.max(dropSideArray2[:,0])
            
            resultLeft  = np.array(np.where(dropSideArray1 == Left))
            resultRight = np.array(np.where(dropSideArray2 == Right))
            
            DeDistance  = Right-Left
            
            DeYlenght = dropSideArray2[resultRight[0,0],1]*(np.ones((1,DeDistance+1)))
            DeXlenght =  np.linspace(Left,Right,DeDistance+1)
            
            print("Dey",DeYlenght[0,0])
            
            ApexOnes = apexFin[0]*np.ones((1,DeDistance+1))
            ApexLine = np.linspace(apexFin[1], apexFin[1]-(DeDistance),DeDistance+1)
            
            upSide1=[]
            
            for i in range(0,Xlimite,1):
                
                if edges[int(ApexLine[-1]),i]==255:
                    
                    np.array([upSide1.append([ApexLine[-1],i])])
                    
                    break
                
            upSide1Array = np.array(upSide1)
            
            upSide2=[]    
            
            for j in range(0,imXsize-10,1):
                
                if edges[int(ApexLine[-1]),Xlimite-j]==255:
                    
                    print(edges[int(ApexLine[-1]),Xlimite-j])
                    
                    np.array([upSide2.append([[ApexLine[-1]],[Xlimite-j]])])
                    
                    break
                
            upSide2Array = np.array(upSide2)
            
            DsDistance = int((upSide2Array[0,1]-upSide1Array[0,1]))+1
            dsOnes = int(ApexLine[-1])*np.ones((1,DsDistance))
            dsLine = np.linspace(int(upSide2Array[0,1]),int(upSide1Array[0,1]),DsDistance)
            
            
            DsReal=DsDistance*ratio
            DeReal=DeDistance*ratio
            DsDeReal = DsReal/DeReal
            
            BondNumber = 0.12836-(0.7577*(DsDeReal))+(1.7713*np.power(DsDeReal,2))-(0.5426*np.power(DsDeReal,3))
            GammaTension = ((Density*np.square(Radio*ratio)*g)/(BondNumber))*1000
            
            
            print("")
            print("------------------------------------------------------------------------------")
            print("An a aproximated surface Tension is" , GammaTension, "[mN/Meter] compared with")
            print("The literature value for water is: 72.75 [nN/M]" )
            print("------------------------------------------------------------------------------")
            print("")
            
            
            
            dl=[]
            rangeValues = []
            
            for i in range(70,90,1):
                
                start =  apexFin[0]-i
                end   =  apexFin[0]+i
                
                np.array([rangeValues.append([start,end])])
                np.array([dl.append([i])])
                
            dlArray  = np.array(dl)    
            rangeArray = np.array(rangeValues)
            RangeSize =  rangeArray[:,0].size
            surfacePoints = []
            Rvalues = []
            centers = []
            Rvalues2 = []
            
            
            for i in range(0,RangeSize,1):
                start2 = rangeArray[i,0]
                end2   = rangeArray[i,1]
                
                surfacePoints = []
                
                for ii in range(start2,end2,1):
                
                    j = dfy-2
#                    j =  imYsize-30
                    
                    while j > 0:
                        
                        if edges[j,ii] == 255:
                            
                            surfacePoints.append([ii,j])
                            sf = np.array(surfacePoints)
                            
                            break
                        
                        j -= 1
                        
                for iii in range(0,1,1):
                    
                    x2 =sf[:,0]
                    y2 =sf[:,1]
                    
                    
                    x2_m = np.mean(x2)
                    y2_m = np.mean(y2)
                
                    x2_mArray = np.ones(x2.size)*x2_m
                    y2_mArray = np.ones(y2.size)*x2_m
                    
                    u2 = x2 - x2_m
                    v2 = y2 - y2_m
                    
                    
                    Suv2  = np.sum(u2*v2)
                    Suu2  = np.sum(u2**2)
                    Svv2  = np.sum(v2**2)
                    Suuv2 = np.sum(u2**2 * v2)
                    Suvv2 = np.sum(u2 * v2**2)
                    Suuu2 = np.sum(u2**3)
                    Svvv2 = np.sum(v2**3)
                
                    A2 = np.array([ [ Suu2, Suv2 ], [Suv2, Svv2]])
                    B2 = np.array([ Suuu2 + Suvv2, Svvv2 + Suuv2 ])/2.0
                
                    uc2, vc2 = np.linalg.solve(A2, B2)
                    xc_2 = x2_m + uc2
                    yc_2 = y2_m + vc2
                
                    Ri_2     = np.sqrt((x2-xc_2)**2 + (y2-yc_2)**2)
                    R_2      = np.mean(Ri_2)
                
                    residu_2 = np.sum((Ri_2-R_2)**2)
                    np.array(Rvalues.append([R_2]))
                    
            RadioArray=np.array(Rvalues)    
            MeanRadio= np.mean(RadioArray)
                    
            
            
            plt.figure()
            plt.title("Radios de curvatura")
            plt.plot(dl,RadioArray,".")
            plt.ylim(182,193)
            plt.grid()
            plt.savefig('Radios de curvatura.png')
            plt.show()
            
                
            Gamma = []
            
            for i in range(0,RangeSize,1):
                
                G = ((Density*np.square(RadioArray[i]*ratio)*g)/(BondNumber))*1000
                np.array([Gamma.append([G])])
                    
            GammaArray = np.array(Gamma)
            meanGamma = np.mean(GammaArray)
            stdGamma  = np.std(GammaArray)
                    
            plt.figure()
            plt.title("Surface Tension Values")
            plt.plot(dl,GammaArray[:,0],".")
            plt.grid()
            plt.ylim(65,78)
            plt.savefig('GraficoTensión.png')
            plt.show()
                    
            print("")
            print("------------------------------------------------------------------------------")
            print(" The Mean Surface Tension  Value is :",meanGamma,"+/-",stdGamma)
            print("------------------------------------------------------------------------------")
            print("")
                    
            plt.figure()
            plt.title("Segmented Droplet")
            plt.imshow(edges,cmap="gray")
            plt.scatter(sf[:,0],sf[:,1])
            plt.scatter(dsLine,dsOnes,color="red")
            plt.scatter(DeXlenght,DeYlenght,color = "orange")
            plt.scatter(xc_2,yc_2,color ="orange")
            plt.scatter(apexFin[0],apexFin[1],color="yellow")
            plt.savefig('segmentedDrop.png')
            plt.show()

            return meanGamma,stdGamma,MeanRadio,BondNumber  #,DeYlenght[0,0]
        
                
                
        GammaValues = getGamma(Rvalues[0],Rvalues[2],size[0],size[1],apex,edges,dropArea)
        print("GammaValues",GammaValues)
        
        
        print("")
        print("--------------------------------------------------------------")
        print("--------------------------------------------------------------")
        
        
        def plotRadio(img,apx,de):
            
            ap= apx
            print(ap)
            de =de
            print(de)
            
            diff = ap[1]-de
            print("difference",diff)
            
            
            cx=ap[0]
            cy=de
            
            print("radio de inicialización",diff)
            s = np.linspace(0, 2*np.pi, 400)
            r =[cy]  + (diff*np.sin(s)/1.0)#  500
            c =[cx]  + (diff*np.cos(s)/1.0) #1
            init = np.array([r, c]).T
            
            plt.figure()
            plt.title("R0 from De")
            plt.imshow(img)
            plt.plot(init[:,1],init[:,0],color="red")
            plt.scatter(cx,cy,color="red")
            plt.show()
            
#        plotRadio(edges,apex,DS[2])
            
        
        def CreateFile(info1,info2,info3,info4,info5):
            
            tension   = np.array([info1])
            deviation = np.array([info2])
            Radio     = np.array([info3])
            Bond      = np.array([info4])
            Nwidth    = np.array([info5])


            data1 = pd.DataFrame({'Mean Surface Tension':tension,'Surface Tension S.Deviation':deviation,'Mean Radio of Curvature':Radio,'Bond Number':Bond,'NeedleWidth px':Nwidth})     
            index = data1.index
            index.name = "Droplets"
            
            print("Data to excel",data1)

            dataToExcel = pd.ExcelWriter("DatosDropets.xlsx",engine="xlsxwriter")
            
            data1.to_excel(dataToExcel,sheet_name='Droplets')
            
            worksheet= dataToExcel.sheets['Droplets']

            worksheet.set_column('A:A',40)
            worksheet.set_column('B:B',26)
            worksheet.set_column('C:C',26)
            worksheet.set_column('D:D',26)
            worksheet.set_column('E:E',26)
            worksheet.set_column('F:F',26)
            
            worksheet.write('A3', 'Segmented Droplet:')
            worksheet.insert_image('B3','segmentedDrop.png',{'x_scale': 0.65, 'y_scale': 0.65})
            worksheet.write('A15', 'Surface Tension Plot:')
            worksheet.insert_image('B15','GraficoTensión.png',{'x_scale': 0.65, 'y_scale': 0.65})
            worksheet.write('A29', 'Radii of curvature Plot:')
            worksheet.insert_image('B26','Radios de curvatura.png',{'x_scale': 0.65, 'y_scale': 0.65})
            
            
            dataToExcel.save()
            
            return dataToExcel.save()
        
        CreateFile(GammaValues[0],GammaValues[1],GammaValues[2],GammaValues[3],Nwidth)
        
        


                
        

#%%
#        
#Method(Coordenadas)


