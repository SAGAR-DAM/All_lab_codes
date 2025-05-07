#Developed by Rakeeb
import pyautogui as pg
from pyfirmata import Arduino, util
import time

board = Arduino('COM15')
print("Communication Successfully started")

'''
#Blinker:
for i in range(20):
    board.digital[13].write(1)
    time.sleep(0.2)  #time in seconds
    board.digital[13].write(0)
    time.sleep(1)
    i+=1


screenWidth, screenHeight = pg.size()
#xc1,yc1=pg.locateCenterOnScreen('Osc_start.png')
#xc2,yc2=pg.locateCenterOnScreen('PI_move.png')
#xc3,yc3=pg.locateCenterOnScreen('labview_start_1.png')

#print(xc1,yc1)

#def move_fn():
	#pg.click(xc1,yc1)
	#pg.click(xc2,yc2)
	#pg.click(xc3,yc3)

time1=time.time()
for i in range(10):
	#a = board.analog[0].read()
	a = board.digital[13].read()
	a = float(a or 0)
	if a >= 0.5:
		print(a)
		continue
	else:	
	#	print("I am high!!")
		#move_fn()
		break
	time.sleep(0.1)
#	print(time.time())
#	print(a)
	i+=1

#move_fn()
time2=time.time()

print(time2 - time1)
'''
it = util.Iterator(board)
it.start()
board.analog[0].enable_reporting()

for i in range(10):
	#a = board.analog[0].read()
	
	a = board.analog[0].read()
	print(a)
	time.sleep(0.1)
