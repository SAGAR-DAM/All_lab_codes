import pyautogui as pg
import numpy as np
import time
import serial as ser
arduino=ser.Serial('COM15',timeout=10000)
xent,yent= pg.locateCenterOnScreen("enter.png")
xtext,ytext= pg.locateCenterOnScreen("tbox.png")
xvel,yvel= pg.locateCenterOnScreen("vbox.png")

vel=float(input("What do you want to set the velocity?"))

arr=np.zeros(20)
for i in range(20):
    arr[i]=(i+1)*50/20

pg.doubleClick(xvel,yvel)
pg.typewrite(str(vel))
lastpos=0.0

def changepos(xtext,ytext,xent,yent):
    global lastpos
    pg.doubleClick(xtext,ytext)
    pg.typewrite(str(arr[c]))
    pg.click(xent,yent)
    time.sleep(float(arr[c]-lastpos)/vel)
    time.sleep(5)
    lastpos=arr[c]

#We consider the stage to be initially at zero
shots = int(input("Please eter the number of shots you want to take"))
delshots = int(input("Please enter the number of shots you want to take at each delay"))
c=0
while c<=shots:
    if c%delshots==0:
        changepos(xtext,ytext,xent,yent)
    arduino.readline()
    c=c+1

