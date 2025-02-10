'''Program to calculate MSE of two images'''
import os
import sys
import cv2 as cv
import numpy as np
from DIPLib import displayImage, histogram, dropLCS, rgb2gray, dropTopLCS

#Determines Mean Square Error for two images
def meanSquareError(imgOne, imgTwo):
    #Extract size information and ensure images are compatible
    imgOneHeight, imgOneWidth = imgOne.shape[:2]
    imgTwoHeight, imgTwoWidth = imgTwo.shape[:2]
    if (imgOneHeight != imgTwoHeight or imgOneWidth != imgTwoWidth):
        print("These photos are not compatible")
        return

    #Declare variable to hold error
    error = 0.0
    
    #Calculate mean square error for every pixel and sum together
    for y in range(imgOneHeight):
        for x in range(imgOneWidth):
            diff = np.subtract(imgOne[y][x][0], imgTwo[y][x][0])
            
            if (diff != 0):            
                error += diff ** 2
            
    #Calculate and return average mean square error for the image
    error /= imgOneHeight * imgOneWidth
    
    return error

#Applies Adaptive Median filter to input image with maximum window size of smax x smax
#Assumes min and max values of 0 and 255, representing salt and pepper noise
def adaptiveMedianFilterSaltPepper(img, smax):
    #Extract image size information
    imgHeight, imgWidth = img.shape[:2]
    
    #Declare new arrays to hold filtered image
    imgFiltered = np.zeros((imgHeight, imgWidth), dtype=np.uint8)
    
    #Declare array of all possible pixel positions and pop those successfully filtered
    heightVals = np.arange(imgHeight)
    widthVals = np.arange(imgWidth)
    remainingPixels = np.array(np.meshgrid(heightVals, widthVals)).T.reshape(-1,2)
    
    #Iterate through and find non-noise pixels
    keep = np.ones(len(remainingPixels), dtype=bool)
    for index, pixel in enumerate(remainingPixels):
        currentPixelVal = img[pixel[0]][pixel[1]][0]
        
        #Place current pixel value in final image if non-noise
        if (currentPixelVal == 0):
            imgFiltered[pixel[0]][pixel[1]] = currentPixelVal
            keep[index] = False
            
    #return imgFiltered
            
    #Remove filtered pixels
    #remainingPixels = remainingPixels[keep]
    
    #Calculate integral histogram of image
    integralHist = integralHistogram(img)
    
    #Declare variable for current window size
    currentWindow = 3
    
    #Apply growing median filter until all pixels are processed or smax is reached
    while(len(remainingPixels) > 0):
        print(currentWindow)
        #Declare mask to pop filtered pixels' locations
        keep = np.ones(len(remainingPixels), dtype=bool)
        
        #Calculate number of pixels needed in each direction for current window size
        currDist = ((currentWindow - 1) / 2) + 1
        
        #Calculate median of window size and place in filtered image if not equivalent to noise
        for index, pixel in enumerate(remainingPixels):
            #Find pixel points for integral hist calculation
            top = int(max(0, pixel[0] - currDist))
            left = int(max(0, pixel[1] - currDist))
            bottom = int(min(imgHeight - 1, pixel[0] + (currDist - 1)))
            right = int(min(imgWidth - 1, pixel[1] + (currDist - 1)))
            
            #Get effective window dimensions for median calculation
            windowX = currentWindow
            windowY = currentWindow
        
            #Get histograms for window corners
            topLeft = integralHist[top][left]
            
            if (top == 0):
                topRight = np.zeros(256)
                topLeft = np.zeros(256)
                windowY = currentWindow + (pixel[0] - currDist)
            else:
                topRight = integralHist[top][right]
                
            if (left == 0):
                bottomLeft = np.zeros(256)
                topLeft = np.zeros(256)
                windowX = currentWindow + (pixel[1] - currDist)
            else:
                bottomLeft = integralHist[bottom][left]
                
            if (right == imgWidth - 1):
                windowX = currentWindow + ((imgWidth - (pixel[1] + 2)) - currDist)
                
            if (bottom == imgHeight - 1):
                windowY = currentWindow + ((imgHeight - (pixel[0] + 2)) - currDist)
                
            bottomRight = integralHist[bottom][right]
            
            #Get histogram of window
            windowHist = getHistFromIntegral(topLeft, topRight, bottomLeft, bottomRight)
            
            #Get median of histogram
            median = int(histogramMedian(windowHist, windowX, windowY))
            
            mini = histogramMin(windowHist)
            maxi = histogramMax(windowHist)
            
            #Add median to filtered picture if not noise
            if ((median > mini and median < maxi) or currentWindow >= smax):
                imgFiltered[pixel[0]][pixel[1]] = median
                
                #Alter result in mask of remaining pixels to keep
                keep[index] = False
        
        #Mask out successfully filtered pixels
        remainingPixels = remainingPixels[keep]
        
        #Increase size of current window
        currentWindow += 2
    
    #Return filtered image
    return imgFiltered
                
        
        
def getHistFromIntegral(topLeft, topRight, bottomLeft, bottomRight):
    #Declare array to hold return histogram
    hist = np.zeros(256, int)
    
    #Iterate through bins and sum values
    for x in range(256):
        hist[x] = bottomRight[x] + topLeft[x] - topRight[x] - bottomLeft[x]
        
    return hist     
            
#Sum overlapping histograms
def sumHist(leftHist, topHist, topLeftHist):
    #Declare array to hold new histogram
    hist = np.zeros(256, int)
    
    #Iterate through bins and sum values
    for x in range(256):
        hist[x] = leftHist[x] + topHist[x] - topLeftHist[x]
        
    return hist
            
#Return integral histogram for input image
def integralHistogram(img):
    #Extract image size information
    imgHeight, imgWidth = img.shape[:2]
    
    #Declare array to hold integral image
    integHist = np.zeros((imgHeight, imgWidth, 256), int)
    
    #Blank histogram for edge pixel integral calculations
    blankHist = np.zeros(256, int)
    
    #Calculate integral histogram for origin pixel
    integHist[0][0][img[0][0][0]] += 1
    
    #Calculate integral histograms for top pixels
    for x in range(1, imgWidth):
        #Sum bounding histograms
        integHist[0][x] = sumHist(integHist[0][x - 1], blankHist, blankHist)
        
        #Increment bin of current pixel's value
        integHist[0][x][img[0][x][0]] += 1
        
    #Calculate integral histograms for left pixels
    for y in range(1, imgHeight):
        #Sum bounding histograms
        integHist[y][0] = sumHist(blankHist, integHist[y - 1][0], blankHist)
        
        #Increment bin of current pixel's value
        integHist[y][0][img[y][0][0]] += 1
        
    #Calculate integral histograms for remaining pixels
    for y in range(1, imgHeight):
        for x in range(1, imgWidth):
            #Sum bounding histograms
            integHist[y][x] = sumHist(integHist[y][x - 1], integHist[y - 1][x], integHist[y - 1][x - 1])
            
            #Increment bin of current pixel's value
            integHist[y][x][img[y][x][0]] += 1
            
    return integHist

#Get median value of histogram
def histogramMedian(histogram, x, y):
    #Calculate number of values in lower half of histogram
    numPixels = (x * y) / 2
    numPixelsInt = int(numPixels)
    
    #Declare variable to hold count of pixels already passed
    count = 0
    
    #Iterate until lower half of pixels passed
    for x in range(256):
        count += histogram[x]
        
        #If count exceeds numPixels, then current x value is median
        if (count >= numPixelsInt):
            #Special case for median pixels in different bins
            if (count + 0.5 == numPixels):
                #Iterate until next occupied bin is found
                y = x + 1
                while(histogram[y] == 0):
                    y += 1
                
                #Return average of median bin values
                return (y + x) / 2
            
            return x
    
#Get max value of histogram
def histogramMax(histogram):
    for x in range(255, -1, -1):
        if (histogram[x] != 0):
            return x
        
#Get minimum value of histogram
def histogramMin(histogram):
    for x in range(0, 256):
        if (histogram[x] != 0):
            return x

#Get paths for input file and bin number
srcDir = os.path.dirname(os.path.abspath(__file__))
inPathOne = str(srcDir + '\\' + sys.argv[1])
inPathTwo = str(srcDir + '\\' + sys.argv[2])
outPath = str(srcDir + '\\' + sys.argv[3])

#Read in source images
imgOne = cv.imread(inPathOne)
imgTwo = cv.imread(inPathTwo)

imgTwo = rgb2gray(imgTwo)

imgTwo = cv.merge((imgTwo, imgTwo, imgTwo))

imgTwo = adaptiveMedianFilterSaltPepper(imgTwo, 11)

imgTwo = cv.merge((imgTwo, imgTwo, imgTwo))

imgTwo = dropTopLCS(imgTwo, 5, 60)

displayImage(imgTwo)

cv.imwrite(outPath, imgTwo)

imgTwo = cv.merge((imgTwo, imgTwo, imgTwo))

#Calculate mean square error
error = meanSquareError(imgOne, imgTwo)
print(error)