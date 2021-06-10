# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 15:38:12 2021

@author: phili
"""

#%%


import cv2
import numpy as np


# ESTA ES LA PRIMERA SCRIPT
#%% VarIables para empezar
##

#InitialImage = cv2.imread("Acmed22.bmp")       ## Reemplaza aca con alguna imagen cualquier para probar esta función
#GrayImage = cv2.cvtColor(InitialImage, cv2.COLOR_BGR2GRAY)



#%%

class CoordinateStore:
    
    
    def __init__(self,image,strings):
        
        self.points= []
        self.image=image
        self.strings = strings
        self.ix,self.iy=0,0
        self.drawing=False
        self.mode=True
        self.font=cv2.FONT_HERSHEY_SIMPLEX
        self.fx,self.fy = 0,0


    def select_point(self,event,x,y,flags,param):
        
            if event == cv2.EVENT_LBUTTONDOWN:
                
                self.pp=[]
                self.points=[]
                self.drawing=True
                self.ix = x
                self.iy=  y
                
                cv2.circle(self.image,(x,y),10,(255,0,0),-1)
                cv2.putText(self.image,"StartingPoint", (x,y), self.font, 1, (255, 0, 0), 1, cv2.LINE_AA)
                
    
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.drawing == True:
                    if self.mode == True:
                        cv2.rectangle(self.image,(self.ix,self.iy),(x,y),(0,255,0),3)
            
            elif event == cv2.EVENT_LBUTTONUP:
                
                self.drawing = False
                if self.mode == True:    
                    
                    self.fx=x
                    self.fy=y
                    
            
                    cv2.rectangle(self.image,(self.ix,self.iy),(self.fx,self.fy),(0,255,0),3)
                    cv2.circle(self.image,(x,y),10,(255,0,0),-1)
                    
                    cv2.putText(self.image,"EndPoint", (x,y), self.font, 1, (255,0, 0), 1, cv2.LINE_AA)
                    self.points.append([self.ix,self.iy,x,y])
                    
                    cv2.rectangle(self.image,(self.ix,self.iy),(self.fx,self.fy),(0,0,255),3)
                    cv2.putText(self.image,"EndPoint", (x,y), self.font, 1, (255,0, 0), 1, cv2.LINE_AA)
                    
                    return self.points





def CallMouseClick(img1,string1):
    
    coordinateStore1=CoordinateStore(img1,string1)
    image   = coordinateStore1.image 
    strings = coordinateStore1.strings
    
    cv2.namedWindow(strings)
    cv2.setMouseCallback(strings,coordinateStore1.select_point)
    
    while(1):
        cv2.imshow(strings ,image)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('m'):
            
            mode = not mode
        elif (k == 27) or (k==13) or (k==32):
            break
    cv2.destroyAllWindows()
    
    return coordinateStore1.points
    



#%%
    

#  CARLOOOSSS !!
    
## Aca yo mando a llamar la función, pero esta se deberia  abrir en la " GeometricGraphicInterface.py"
    
#ReturnMouseClicks =np.asarray(CallMouseClick(InitialImage,"Select drop Region"))
#    
#
#print("return of the pointsss",ReturnMouseClicks)
#
#
#



#%%
