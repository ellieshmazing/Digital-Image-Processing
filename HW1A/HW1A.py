import cv2 as cv
import numpy as np

#Create string for base path to image folder
path = r'C:\Users\ellie\OneDrive\Desktop\Fall 2024\Digital Image Processing\HW1A\\'
path = path[:-1]

#Read in source image and split into color channels
img = cv.imread(r'C:\Users\ellie\OneDrive\Desktop\Fall 2024\Digital Image Processing\HW1A\Bone_cells.jpg')
imgB, imgG, imgR = cv.split(img)

'''Color to Greyscale Conversion'''
#Weigh channels to form greyscale image
imgGrey = cv.addWeighted(imgB, .1626, imgG, .8374, 0)
imgGrey = cv.addWeighted(imgGrey, .7011, imgR, .2989, 0)

#Display image
cv.imshow("Cells", imgGrey)
cv.waitKey(0)
cv.destroyAllWindows()

#Save greyscale image
filename = str(path + "Bone_cells_grey.jpg")
cv.imwrite(filename, imgGrey)

'''Greyscale Image Binarization'''
#Get user input for threshold value
threshold = input("What intensity level do you want to set the threshold at (0-255)? ")

#Apply threshold function and display image
ret, imgGreyThreshold = cv.threshold(imgGrey, float(threshold), float(255), cv.THRESH_BINARY)
cv.imshow("Cells", imgGreyThreshold)
cv.waitKey(0)
cv.destroyAllWindows()

#Save greyscale binarization image
filename = str(path + "Bone_cells_threshold_grey.jpg")
cv.imwrite(filename, imgGreyThreshold)

'''Color Channel Binarization'''
#Get user input for threshold value
threshold = input("What intensity level do you want to set the blue threshold at (0-255)? ")

#Apply threshold and display image
ret, imgBlueThreshold = cv.threshold(imgB, float(threshold), float(255), cv.THRESH_BINARY)
cv.imshow("Cells", imgBlueThreshold)
cv.waitKey(0)
cv.destroyAllWindows()

#Save greyscale binarization image
filename = str(path + "Bone_cells_threshold_blue.jpg")
cv.imwrite(filename, imgBlueThreshold)