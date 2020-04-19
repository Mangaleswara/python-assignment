Python assignment:
"""
GUI CREATED WITH START, CAPORT, STOP
Caport captures the image and for temperature csv values suitable for export along with capture of normal image
"""
import tkinter as tk 
import time 
import board 
import busio 
import board 
import adafruit_amg88xx 
from Adafruit_AMG88xx import Adafruit_AMG88xx 
import pandas as pd 
import pygame 
import os 
import math 
import bluetooth 
import numpy as np 
from scipy.interpolate import griddata

from colour import Color
import picamera 
win= tk.Tk()
win.title("Camera")
on=True
def sensor():
	global on
	on=True 
	#low range of the sensor (this will be blue on the screen)
	MINTEMP = 28

#high range of the sensor (this will be red on the screen)
	MAXTEMP = 34

#how many color values we can have
	COLORDEPTH = 1024

	os.putenv('SDL_FBDEV', '/dev/fb1')
	pygame.init()

#initialize the sensor
	sensor = Adafruit_AMG88xx()

	points = [(math.floor(ix / 8), (ix % 8)) for ix in range(0, 64)]
	grid_x, grid_y = np.mgrid[0:7:32j, 0:7:32j]

#sensor is an 8x8 grid so lets do a square
	height = 720
	width = 720

#the list of colors we can choose from
	blue = Color("indigo")
	colors = list(blue.range_to(Color("red"), COLORDEPTH))

#create the array of colors
	colors = [(int(c.red * 255), int(c.green * 255), int(c.blue * 255)) for c in colors]

	displayPixelWidth = width / 30
	displayPixelHeight = height / 30

	lcd = pygame.display.set_mode((width, height))

	lcd.fill((255,0,0))

	pygame.display.update()
	pygame.mouse.set_visible(False)

	lcd.fill((0,0,0))
	pygame.display.update()

#some utility functions
	def constrain(val, min_val, max_val):
    		return min(max_val, max(min_val, val))

	def map(x, in_min, in_max, out_min, out_max):
  		return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

#let the sensor initialize
	time.sleep(.1)
	x=100
	while (x):

		#read the pixels
		pixels = sensor.readPixels()
		pixels = [map(p, MINTEMP, MAXTEMP, 0, COLORDEPTH - 1) for p in pixels]
		
		#perdorm interpolation
		bicubic = griddata(points, pixels, (grid_x, grid_y), method='cubic')
		
		#draw everything
		for ix, row in enumerate(bicubic):
			for jx, pixel in enumerate(row):
				pygame.draw.rect(lcd, colors[constrain(int(pixel), 0, COLORDEPTH- 1)], (displayPixelHeight * ix, displayPixelWidth * jx, displayPixelHeight, displayPixelWidth))
		
		pygame.display.update()
		x=x-1
	#win.update(10)
def export():
	global on
	on=True
	i2c = busio.I2C(board.SCL, board.SDA)
	amg = adafruit_amg88xx.AMG88XX(i2c)
	time.sleep(1)
	# Create an empty list and append row temperature values
	lst=[] 
	for row in amg.pixels:
		lst.append(row)
	df = pd.DataFrame(lst)
	print (df)
	df.to_csv('sid.csv')
	# normal image capturing
	camera=picamera.PiCamera()
	camera.capture('normal.jpg')
	#IR image capture
		MINTEMP = 28

	MAXTEMP = 34


	COLORDEPTH = 1024

	os.putenv('SDL_FBDEV', '/dev/fb1')
	pygame.init()

	sensor = Adafruit_AMG88xx()

	points = [(math.floor(ix / 8), (ix % 8)) for ix in range(0, 64)]
	grid_x, grid_y = np.mgrid[0:7:32j, 0:7:32j]

	height = 720
	width = 720

	blue = Color("indigo")
	colors = list(blue.range_to(Color("red"), COLORDEPTH))

	colors = [(int(c.red * 255), int(c.green * 255), int(c.blue * 255)) for c in colors]

	displayPixelWidth = width / 30
	displayPixelHeight = height / 30

	lcd = pygame.display.set_mode((width, height))

	lcd.fill((255,0,0))

	pygame.display.update()
	pygame.mouse.set_visible(False)

	lcd.fill((0,0,0))
	pygame.display.update()

	def constrain(val, min_val, max_val):
    		return min(max_val, max(min_val, val))

	def map(x, in_min, in_max, out_min, out_max):
  		return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

	time.sleep(.1)
	if on:
		pixels = sensor.readPixels()
		pixels = [map(p, MINTEMP, MAXTEMP, 0, COLORDEPTH - 1) for p in pixels]
	
			bicubic = griddata(points, pixels, (grid_x, grid_y), method='cubic')
	
			for ix, row in enumerate(bicubic):
			for jx, pixel in enumerate(row):
				pygame.draw.rect(lcd, colors[constrain(int(pixel), 0, COLORDEPTH- 1)], (displayPixelHeight * ix, displayPixelWidth * jx, displayPixelHeight, displayPixelWidth))
	
		pygame.display.update()
		pygame.image.save(lcd, "thermal image.jpeg")	


def off():
	global on
	on=False
	print("sensor is switched off")
	win.quit()
button = tk.Button(win, text='Start', width=25, command=sensor) 
button.pack() 
button = tk.Button(win, text='Caport', width=25, command=export) 
button.pack() 
button = tk.Button(win, text='Stop', width=25, command=off) 
button.pack() 
tk.mainloop() 
"""
Image enhancement by sharpening
"""
from PIL import Image
from PIL import ImageFilter
import cv2

# Open an already existing image
imageObject = Image.open("thermal image.jpeg");
imageObject.show();

# Apply sharp filter
sharpened1 = imageObject.filter(ImageFilter.SHARPEN);
sharpened2 = sharpened1.filter(ImageFilter.SHARPEN);
# Show the sharpened images
sharpened1.show();
sharpened2.show();
cv2.imwrite('sharpened img.png',sharpened2)
"""
K means clustering and Optical surface temperature of entire eye feature
"""
from PIL import Image
import colorsys
import matplotlib.pyplot as plt
import numpy
import cv2
original_image = cv2.imread('sharpened img.png')# read an image
img=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)# convert it from bgr to rgb 
vectorized = img.reshape((-1,3))
vectorized = np.float32(vectorized)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 7
attempts=10
ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
result_image = res.reshape((img.shape))
figure_size = 15
result_image=cv2.cvtColor(result_image,cv2.COLOR_BGR2RGB)
cv2.imwrite('segment ir.png',result_image)
plt.figure(figsize=(figure_size,figure_size))
plt.subplot(1,2,1),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(result_image)
plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
plt.show()

im = Image.open('segment ir.png')

NUM_BUCKETS = 6 # 6 main colors are used
colour_counts = [0] * NUM_BUCKETS

for pixel in im.getdata():
    hue, saturation, value = colorsys.hsv_to_rgb(pixel[0], pixel[1], pixel[2])
    hue_bucket = hue * NUM_BUCKETS // 255 # Using python3 to get an int
    colour_counts[hue_bucket] += 1

colour_names = ["red", "yellow", "green", "cyan", "blue", "magenta"]
for name, count in [x for x in zip(colour_names, colour_counts)]:
    print("{n} = {c}".format(n=name, c=count))
    plt.imshow(im)
# separate foreground from background
backg = 0
real = 0

for pixel in im.getdata():
    if pixel == (0,0,0 ): # if your image is RGB (if RGBA, (0, 0, 0, 255) or so
        backg += 1
    else:
        real += 1
print('backg=' + str(backg)+', fore='+str(real))
plt.imshow(im)
# for corresponding colors given above in gui, backtracing temperature values from color image
b=int(31.8*colour_counts[4])
g=int(32.1*colour_counts[2])
y=int(32.4*colour_counts[1])
r=int(33*colour_counts[0])
sum=b+g+y+r
print("sum=",sum)
div= int(colour_counts[4]+colour_counts[2]+colour_counts[1]+colour_counts[0])
print("divident=",div)
avg_temp=sum/div
print("avg temperature=",avg_temp)
plt.imshow(im)

"""
Feature2- Application of cornea mask and Cornea Optical Surface Temperature
"""
from PIL import Image
import colorsys
import matplotlib.pyplot as plt
import cv2
import numpy as np
im=cv2.imread('segment ir.png') #read an image
# A mask is created for all images with radius 25% of entire eye, and temperature of cornea part alone is calculated 
# creation of mask
h, w = im.shape[:2]
Y, X = np.ogrid[:h, :w]
center = (int(w/2), int(h/2))
H=int(h/2)
radius=w/4
dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
mask = dist_from_center >= radius
masked_im = im.copy()
masked_im[mask]=0
#plt.imshow(masked_im)
a=cv2.imwrite('cornea_sept.png',masked_im)
im = Image.open('cornea_sept.png')

NUM_BUCKETS = 6 # 6 basic colors used
colour_counts = [0] * NUM_BUCKETS

for pixel in im.getdata():
    hue, saturation, value = colorsys.hsv_to_rgb(pixel[0], pixel[1], pixel[2])
    hue_bucket = hue * NUM_BUCKETS // 255 # Using python3 to get an int
    colour_counts[hue_bucket] += 1

colour_names = ["red", "yellow", "green", "cyan", "blue", "magenta"]
for name, count in [x for x in zip(colour_names, colour_counts)]:
    print("{n} = {c}".format(n=name, c=count))
masked_im=cv2.cvtColor(masked_im, cv2.COLOR_BGR2RGB)
plt.imshow(masked_im)    
b1=int(31.8*colour_counts[4])
g1=int(32.1*colour_counts[2])
y1=int(32.4*colour_counts[1])
r1=int(33*colour_counts[0])
sum1=b1+g1+y1+r1
print("sum=",sum)
div1= int(colour_counts[4]+colour_counts[2]+colour_counts[1]+colour_counts[0])
print("divident=",div)
avg_temp1=sum1/div1
print("cornea temperature=",avg_temp1)
"""
Feature 3- Temeprature Deviation along Cornea
"""
from PIL import Image
import colorsys
import matplotlib.pyplot as plt
import cv2
import numpy as np
im = cv2.imread('segment ir.png')
h, w = im.shape[:2]
Y, X = np.ogrid[:h, :w]
print(h,w)
center = (int(w/2), int(h/2))
H=int(h/2)
radius=w/4
dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
#print(dist_from_center)
mask = dist_from_center >= radius
masked_im = im.copy()
l=masked_im[mask]=0
plt.imshow(masked_im)
f,g=im.shape[:2]
print(f,g)
#plt.imshow(d)
masked_im=cv2.cvtColor(masked_im, cv2.COLOR_BGR2RGB)
# form a rectangle image along centre
image = cv2.rectangle(masked_im, (27,H), (81,H),(255,0,0),-1)
image_data = np.asarray(masked_im)
count=0
for i in range(H-1,H):
    for j in range(27,81):
           print(image_data[i][j])
           if image_data[i,j,0]==image_data[i+1,j+1,0] and image_data[i,j,1]==image_data[i+1,j+1,1] and image_data[i,j,2]==image_data[i+1,j+1,2]:
              count=count+0
           else:
               count=count+0.3
count=count/4
print('tdc=',count)
              

plt.imshow(image_data)
image_data=cv2.cvtColor(masked_im, cv2.COLOR_BGR2RGB)
aos=cv2.imwrite('tdc.png',image_data)

"""
Checking using vibgyor segmentation
"""
import numpy as np
import PIL as Image
import cv2
import matplotlib.pyplot as plt
import sketcher 
import cv2
image=a=cv2.imread('cornea circle.png')
plt.imshow(a)
a=cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
b=a.shape[2]
if b==3:
    print('valid image')
else:
    print('select a valid rgb image')
C = np.zeros(shape=image.shape)
if np.max(image) > 1:
    GL = 255
else:
    GL = 1
GL=255
f1, f2, f3 = (GL * 1), (GL * 0.6), (GL * 0.68)
f4, f5, f6 = (GL * 0), (GL * 1), (GL * 0.6)

def cfilter(image, f1, f2, f3, f4, f5, f6, m, flg):
    C = np.zeros(shape=image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if flg == 0:
                if f2 <= image[i, j, 0] <= f1 and f4 <= image[i, j, 1] <= f3 and f6 <= image[i, j, 2] <= f5 and image[i, j, m-1] == np.max(image[i, j]):
                    C[i, j] = image[i, j]
                else:
                    C[i, j] = image[i, j, 0] * 0.3 + image[
                        i, j, 1] * 0.59 + image[i, j, 2] * 0.11
            else:
                if f2 <= image[i, j, 0] <= f1 and f4 <= image[i, j, 1] <= f3 and f6 <= image[i, j, 2] <= f5:
                    C[i, j] = image[i, j]
                else:
                    C[i, j] = image[i, j, 0] * 0.3 + image[
                        i, j, 1] * 0.59 + image[i, j, 2] * 0.11
    return C
C = cfilter(image, f1, f2, f3, f4, f5, f6, 3, 1)/GL
D=np.asarray(C)
print(D)
plt.imshow(D)

