# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 14:29:01 2021

@author: phili
"""


#%% Abrir Interfaz Grafica
# que a su  vez llama la funcion de Guardar Los datos del Mouse
# ESTA ES LA SEGUNDA SCRIPT


import tkinter as tk
from tkinter import *
from tkinter import filedialog
#%%

import MouseClick   ## 

#%%
import numpy as np
import cv2


#%%

class GraphicInterface:
    
  def __init__(self, master):

#      
      frame =Frame(master)
      frame.pack()
      self.Drop = []
      self.Need = []
      self.Image=[]
      self.Dens1=[]
      self.Dens2=[]
      self.Thick=[]


      
      myLabel = Label(frame,text= "Insert Data")
      myLabel.pack(padx=10,pady=10)
      
      
      frame =LabelFrame(frame,text="Parameters",padx=20,pady=20)

      
      #%%
             
      self.myLabel1 = Label(frame,text= "Give Drop Density(Kg/m3):")
      self.myLabel1.pack(padx=10,pady=0)
      
      self.Entry1 = Entry(frame,width=30, bd=1)
      self.Entry1.pack(pady=10)
      self.Entry1.insert(0,float(1000))
      
      self.myLabel2 = Label(frame,text= "Give Medium Density(kg/m3):")
      self.myLabel2.pack(padx=10,pady=0)
      
      self.Entry2 = Entry(frame,width=30, bd=1)
      self.Entry2.pack(pady=10)
      self.Entry2.insert(0,float(1002.9))
      
      self.myLabel3 = Label(frame,text= "Give Needle Thickness(mm):")
      self.myLabel3.pack(padx=10,pady=0)
      
      self.Entry3 = Entry(frame,width=30, bd=1)
      self.Entry3.pack(pady=10)
      self.Entry3.insert(0,float(0.7176))
      
      frame.pack(padx=100,pady=50)

            


      
      
      def CallBox():
    
          frame.filename = filedialog.askopenfilename(initialdir = "Desktop",title = "Select Image",filetypes = ( ("all files","*.*") ,("bmp files","*.bmp"), ("jpeg files","*.jpg"), ("png files","*.png")  ))
          img = cv2.imread(frame.filename)
          img_Temp = img.copy()
          img_Temp2 = img.copy()
          
          
          self.Drop  = np.array(MouseClick.CallMouseClick(img_Temp, "Select drop Region And Press Enter"))   # ESTAS COORDENADAS DEBO GUARDAR
          self.Need  = np.array(MouseClick.CallMouseClick(img_Temp2,"Select Needle Region And Press Enter")) 
          self.Image=img
          
          self.Dens1= float(self.Entry1.get())
          self.Dens2= float(self.Entry2.get())
          self.Thick= float(self.Entry3.get())

          
        
          print("Drop's Density:",self.Dens1)
          print("Needle's Density:",self.Dens2)
          print("Needle's Thickness:",self.Thick)          
          
          
          
          
          print("Drop Coordinates:", self.Drop)
          print("Needle Coordinates",self.Need)

          return self.Drop,self.Need,self.Image,self.Dens1,self.Dens2,self.Thick


      frame =LabelFrame(frame,text="Search for Image",padx=20,pady=20)
      frame.pack(padx=20,pady=20)
      
      
      self.button = Button(frame,text="Open Image", fg="black",command=CallBox)
      self.button.pack(padx=10,pady=20)   
      
      
      
      
    

#%%
      
#%%    
      
def mainWindow():
    
    root = tk.Tk()
    root.title("SCIAN-Lab's Drop Surface Tension Measurement Tool ")
    root.minsize(400, 100)
    app =GraphicInterface(root)

    root.mainloop()  
    
    DropCoords= np.array(app.Drop)
    NeedCoords=np.array(app.Need)
    imageT =app.Image
    
    Dens1 =app.Dens1
    Dens2 =app.Dens2
    Thick  =app.Thick

#    print("cors",DropCoords)
    
    return DropCoords, NeedCoords, imageT,Dens1,Dens2,Thick
    
#hh = mainWindow()

#%%
 

    