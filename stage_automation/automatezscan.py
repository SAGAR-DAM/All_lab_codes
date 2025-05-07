#Code created by Niladri and Rakeeb
import pyautogui as pg
import time


screenWidth, screenHeight = pg.size()
xc2,yc2=pg.locateCenterOnScreen('sc3.png')
print(xc2,yc2)
time1=0.0

chlr =int(input("0 for left and 1 for right:"))
if chlr == 0:
    xc3,yc3=pg.locateCenterOnScreen('sc4.png')
else:
    xc3,yc3=pg.locateCenterOnScreen('sc2.png')
ch=int(input("Do u wanna start?(1/0)\n"));
if ch == 1:
    pg.click(xc2,yc2)
    pg.click(xc3,yc3)
    time1 = time.time()
    
xc6,yc6=pg.locateCenterOnScreen('sc6.png')
pg.click(xc6,yc6)
xc5,yc5=pg.locateCenterOnScreen('sc5.png')
time4=time.time()
while(1):
    if time.time()-time1>=25:
        pg.click(xc5,yc5)
        break
