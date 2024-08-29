import cv2 as cv
import numpy as np

#Create string for base path to image folder
path = r'C:\Users\ellie\OneDrive\Desktop\Fall 2024\Digital Image Processing\HW1A\\'
path = path[:-1]

#Read in source image and split into color channels
img = cv.imread(r'C:\Users\ellie\OneDrive\Desktop\Fall 2024\Digital Image Processing\HW1A\Bone_cells.jpg')
imgB, imgG, imgR = cv.split(img)

#Weigh channels to form greyscale image
imgGrey = cv.addWeighted(imgB, .1140, imgG, .5871, 0)
imgGrey = cv.addWeighted(imgGrey, .7011, imgR, .2989, 0)

#Display image
cv.imshow("Cells", imgGrey)
cv.waitKey(0)

#Save greyscale image
filename = str(path + "Bone_cells_grey.jpg")
cv.imwrite(filename, imgGrey)
