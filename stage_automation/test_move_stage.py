#Developed by Rakeeb
import pyautogui as pg
import time

screenWidth, screenHeight = pg.size()
xc1,yc1=pg.locateCenterOnScreen('Osc_start.png')
xc2,yc2=pg.locateCenterOnScreen('PI_move.png')
xc3,yc3=pg.locateCenterOnScreen('labview_start_1.png')

#print(xc1,yc1)

pg.click(xc1,yc1)
pg.click(xc3,yc3)
pg.click(xc2,yc2)

print(xc1,yc1)
print(xc2,yc2)
print(xc2,yc2)

