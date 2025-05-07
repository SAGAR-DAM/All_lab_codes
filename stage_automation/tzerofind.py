import pyautogui as pg
import numpy as np
import time
import serial as ser
arduino=ser.Serial('COM16',timeout=10000)
#xc2,yc2=pg.locateCenterOnScreen('sc3.png')
#pg.click(xc2,yc2)
xent,yent= pg.locateCenterOnScreen("enter.png")
xtext,ytext= pg.locateCenterOnScreen("tbox.png")
xvel,yvel= pg.locateCenterOnScreen("vbox.png")

vel=float(input("What do you want to set the velocity?"))

target=float(input("What is the target you want to go to? "))

pg.doubleClick(xvel,yvel)
pg.typewrite(str(vel))


def changepos(xtext,ytext,xent,yent,target):
    pg.doubleClick(xtext,ytext)
    pg.typewrite(str(target))
    arduino.readline()
    pg.click(xent,yent)

changepos(xtext,ytext,xent,yent,target)
