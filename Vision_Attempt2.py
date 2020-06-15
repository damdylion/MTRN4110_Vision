import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
import serial
import time
import argparse
import imutils
from collections import deque
from imutils.video import VideoStream


def startCamera():
# =============================================================================
#     For camera testing (ignore above code, it's fucked)
#    
#    Uncomment line:
#       77: when connected to maze camera
#       78: when you want to use laptop camera
#
# =============================================================================
    
#    usbCamera = cv2.VideoCapture(cv2.CAP_DSHOW)
##    usbCamera = cv2.VideoCapture(0)
#    
#    usbCamera.set(cv2.CAP_PROP_FPS, 30)
#    usbCamera.set(cv2.CAP_PROP_FRAME_WIDTH,800)
#    usbCamera.set(cv2.CAP_PROP_FRAME_HEIGHT,500)
#    usbCamera.set(3,1920)
#    usbCamera.set(4,1080)
#    
#    if usbCamera.isOpened(): # try to get the first frame
#            rval, frame = usbCamera.read()
#    else:
#            rval = False
#    while rval:
#        cv2.imshow("USB Camera", frame)
#        rval, frame = usbCamera.read()
#        key = cv2.waitKey(20)
#        if key == 27: # exit on ESC
#            usbCamera.release()    
#            vs = VideoStream(src=1,resolution = (1920,1080),framerate = 30).start()
#            cv2.destroyWindow("USB Camera")
#            return frame,vs


# =============================================================================
# Comment out if connected to camera
# =============================================================================
    
    picture = cv2.imread('fcam6.jpg')
#    vs = VideoStream(src=1,resolution = (1920,1080),framerate = 30).start()
#    picture = cv2.imread('car2.jpg')
    return picture

def startVideoStream():
    vs = VideoStream(src=1,resolution = (1920,1080),framerate = 30).start()
    return vs


def displayPicture(picture):
    cv2.imshow("Snap",picture)    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def testProcess(picture,gray,edges,red):
# This function tests the edge detection next to the origintal image.
    
    picture = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
    red = cv2.cvtColor(red, cv2.COLOR_BGR2RGB)
    images = [picture,gray,edges,red]
    for i in range(4):
        plt.subplot(1,4,i+1)
        plt.imshow(images[i],'gray')
        
def redOverLay(picture):        
    
    red = copy.deepcopy(picture)
    gray = cv2.cvtColor(picture,cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.blur(gray,(5,5))
    edges = cv2.Canny(blur_gray,60,100)        
    lines = cv2.HoughLinesP(edges,1,np.pi/360,50,70)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(red, (x1, y1), (x2, y2), (0, 255, 0), 30)
#    cv2.imshow("test",red)
    vertIndex = [25,221,431,631,838,1052,1266,1474,1681,1878]
    horIndex = [42,241,448,656,857,1062]
    horList = []
    vertList = []
#    cv2.imshow("test",red)
#   Checking for horizontal walls.
    for x in horIndex:
#   Iterates through the 6 horizontal walls     
#        print("X is:",x)
#        print(len(vertIndex))
#        dick = 0
        for c in range(len(vertIndex)):
#   Iterates through the vertical walls to check    
            if c == 9:
                break
            else:
                startH = vertIndex[c]
                endH = vertIndex[c+1]
#                print("Start is:",start)
#                print("End is:", end)
#            print(c)    
            a = startH
#            print("A is :",a)
            count = 0
#            print("Before loop:",count)
            for a in range(startH,endH):
                if red[x,a,1] == 255:
                    count+=1
#                    print(count)
                else:
                    continue
            if count > 170:
                cv2.line(picture,(endH,x),(startH,x),(0,0,255),10)
#                print(dick)
#                dick+=1
                horList.append("1")
            else:
                horList.append("0")
                continue
#            print("After for loop:", count)
            
#   Vertical Wall Checker
    for h in vertIndex:
        for d in range(len(horIndex)):
            if d == 5:
                break
            else:
                startV = horIndex[d]
                endV = horIndex[d+1]              
            count = 0
            for g in range(startV,endV):
                if red[g,h,1] == 255:
                    count+=1
                else:
                    continue
            if count > 170:
                cv2.line(picture,(h,endV),(h,startV),(0,0,255),10)
                vertList.append("1")
            else:
                vertList.append("0")
                continue

#    cv2.imshow("Test",picture)
#    cv2.imshow("Test1",red)
#    print(vertList)
#    print("Horizontal Matrix is: ",horMat)
#    print("Vertical Matrix is: ", vertList)                
    return picture, horList,vertList

def reArrangeVertList(vertList):
    
#    horMat = [[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]
    vertMat = [[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]]
#    horIndex = 0
    vertIndex = 0
#   converting vertList to vertMat
    for i in range(len(vertList)):
#        print("i is:",i)
#        print("mod is:" ,i%5)
        if vertList[i] == '1':
            vertMat[i%5][vertIndex] = 1
        if i%5 == 4:
            vertIndex+=1
#            print("index is: ",vertIndex)
        else:
            continue
#    for i in range(len(horList)):
    vertIndex = 0
    newVert = []
    for i in range(50):
        if vertMat[vertIndex][i%10] == 1:
            newVert.append("1")
        else:
            newVert.append("0")
        if i%10 == 9:
            vertIndex+=1
        
#    print(newVert) 
#    print("Horizontal Matrix is: ",horMat)
#    print("Vertical List is: ",vertList)
#    print("Vertical Matrix is: ", vertMat)
    return newVert

def findCar(picture):
    
    binImage = cv2.cvtColor(picture,cv2.COLOR_BGR2GRAY)
    cellLocation = []
# =============================================================================
# Detection of the car
#    
# We know that the car will be positioned in one of the four corners, therefore
# we will chech the 4 corners.   
#    
#    
#   https://github.com/michael-pacheco/opencv-arrow-detection/blob/master
#   /arrow_detection_hough_lines.py
# =============================================================================
    
# =============================================================================
#   ROI Regions
# =============================================================================
#   Top left corner
#    roi1 = binImage[70:225,50:250]
#    roi1 = binImage[20:300,20:350]
#   Top right corner 
#    roi2 = binImage[20:225,1610:1820]
#   Bottom left corner 
    roi3 = binImage[875:1050,50:250]
#   Bottom right corner 
    roi4 = binImage[875:1050,1610:1820]
#    roi4 = binImage[900:1000,1660:1750]
#    roi1 = binImage[125:225,150:300]
    
    cv2.imshow("ROI",roi4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# =============================================================================
#   Setting Up the Blob Detection
# =============================================================================
    params = cv2.SimpleBlobDetector_Params()
#    params.minThreshold = 0
#    params.maxThreshold = 2000
    params.filterByArea = True
    params.minArea = 250
    params.maxArea = 10000
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.filterByConvexity = True
    params.minConvexity = 0.5
    params.filterByInertia = True
    params.minInertiaRatio = 0.5
#    params.filterByColor = True    
#    params.blobColor = 255
    detector = cv2.SimpleBlobDetector_create(params)
    
# =============================================================================
#   Checking each region for the car
# =============================================================================
    
#    carDetect = detector.detect(roi1)
#    if carDetect != []:
#        print("Car Detected in Top Left Corner")
#        cellLocation.append("00")      
#        return cellLocation,roi1
#    
#    carDetect = detector.detect(roi2)
#    if carDetect != []:
#        print("Car Detected in Top Right Corner")
#        cellLocation.append("08")      
#        return cellLocation,roi2
 
    carDetect = detector.detect(roi3)
    if carDetect != []:
        print("Car Detected in Bottom Left Corner")
        cellLocation.append("36")
        return cellLocation,roi3
    
    carDetect = detector.detect(roi4)
    if carDetect != []:
        print("Car Detected in Bottom Right Corner")
        cellLocation.append("44")      
        return cellLocation,roi4
    
    print("No Car Detected, you're trash")
    
    
    
    overLayedCar = cv2.drawKeypoints(picture,carDetect,np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("yeet",overLayedCar)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    return cellLocation

def getOrientation(arrow):
    
# =============================================================================
# Detection of the orientation of the car base on an arrow    
# =============================================================================
    orientation = []
#    cv2.imwrite("roiCal.jpg",arrow)
#    gray = cv2.cvtColor(arrow,cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(arrow,3,0.1,50)
    corners = np.int0(corners)
    color = cv2.cvtColor(arrow,cv2.COLOR_GRAY2RGB)

    i = 1
    point1 = []
    point2 = []
    point3 = []

#    cv2.imshow("yeet",color)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()


    for corner in corners:
        x,y = corner.ravel()
#    cv2.circle(arrow,(x,y),5,(0,0,255),10)
    
    
        if i == 1:
#        point1 = unumpy.uarray([corner[0][0],corner[0][1]],[5,5])
            point1 = corner
#        p1 = unumpy.uarray([corner[0][0],corner[0][1]],[5,5])
            p1_upper = point1+5
            p1_lower = point1-5
#        print(p1_lower[0][0])
#        print(p1_upper[0][0])
            p1_rangeX = np.arange(p1_lower[0][0],p1_upper[0][0],1)
            p1_rangeY = np.arange(p1_lower[0][1],p1_upper[0][1],1)
#        p1_full = np.uint8(p1)
#        print(p1_rangeY)
#        RED
            cv2.circle(color,(point1[0][0],point1[0][1]),5,(0,0,255),10)
            i+=1
        elif i == 2:
            point2 = corner
            p2_lower = point2-5
            p2_upper = point2+5
            p2_rangeX = np.arange(p2_lower[0][0],p2_upper[0][0],1)
            p2_rangeY = np.arange(p2_lower[0][1],p2_upper[0][1],1)        
#        p2 = unumpy.uarray([corner[0][0],corner[0][1]],[5,5])  
#        BLUE
            cv2.circle(color,(point2[0][0],point2[0][1]),5,(255,0,0),10)
            i+=1
        else:
            point3 = corner
            p3_lower = point3-5
            p3_upper = point3+5
            p3_rangeX = np.arange(p3_lower[0][0],p3_upper[0][0],1)
            p3_rangeY = np.arange(p3_lower[0][1],p3_upper[0][1],1)        
#        p3 = unumpy.uarray([corner[0][0],corner[0][1]],[5,5])
#        GREEN
            cv2.circle(color,(point3[0][0],point3[0][1]),5,(0,255,0),10)
                
    cv2.imshow("Color Checker",color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #print(p1)
    #print(corners)
    print(point1)
    print(point2)
    print(point3)
        
#    if point1[0][0] == point2[0][0] or point2[0][0] == point3[0][0] or point3[0][0] == point1[0][0]:
##    if p1 == p2 or p2 == p3 or p3 == p1:
##            print("Arrow is left or right")
#        if point1[0][0] > point2[0][0] and point1[0][0] > point3[0][0] or point2[0][0] > point3[0][0] and point2[0][0] > point1[0][0] or point3[0][0] > point1[0][0] and point3[0][0] > point2[0][0]:
#            print("Right Arrow")
#            orientation.append("1")
#        elif point1[0][0] < point2[0][0] and point1[0][0] < point3[0][0] or point2[0][0] < point3[0][0] and point2[0][0] < point1[0][0] or point3[0][0] < point1[0][0] and point3[0][0] < point2[0][0]:
#            print("Left Arrow")
#            orientation.append("3")
#    elif point1[0][1] == point2[0][1] or point2[0][1] == point3[0][1] or point3[0][1] == point1[0][1]:       
##        print("Up or down arrow")
#        if point1[0][1] > point2[0][1] and point1[0][1] > point3[0][1] or point2[0][1] > point3[0][1] and point2[0][1] > point1[0][1] or point3[0][1] > point1[0][1] and point3[0][1] > point2[0][1]:
#            print("Down Arrow")
#            orientation.append("2")
#        elif point1[0][1] < point2[0][1] and point1[0][1] < point3[0][1] or point2[0][1] < point3[0][1] and point2[0][1] < point1[0][1] or point3[0][1] < point1[0][1] and point3[0][1] < point2[0][1]:
#            print("Up Arrow")
#            orientation.append("0")
    if any(np.in1d(p1_rangeX,p2_rangeX)) == True or any(np.in1d(p2_rangeX,p3_rangeX)) == True or any(np.in1d(p3_rangeX,p1_rangeX)) == True:
        if(any(np.in1d(p1_rangeX,p2_rangeX))) == True:
#        Established that point 1 and 2 are the base.
            referenceX = (point1[0][0]+point2[0][0])/2
            if(point3[0][0] > referenceX):
                print("Right Arrow")
                orientation.append("1")
            else:
                print("Left Arrow")
                orientation.append("3")
        elif(any(np.in1d(p2_rangeX,p3_rangeX))) == True:
#        Established that point 2 and 3 are the base.
            referenceX = (point2[0][0]+point3[0][0])/2
            if(point1[0][0] > referenceX):
                print("Right Arrow")
                orientation.append("1")
            else:
                print("Left Arrow")
                orientation.append("3")
        else:
#        Established that point 2 and 3 are the base.
            referenceX = (point3[0][0]+point1[0][0])/2
            if(point2[0][0] > referenceX):
                print("Right Arrow")
                orientation.append("1")
            else:
                print("Left Arrow")
                orientation.append("3")
    elif any(np.in1d(p1_rangeY,p2_rangeY)) == True or any(np.in1d(p2_rangeY,p3_rangeY)) == True or any(np.in1d(p3_rangeY,p1_rangeY)) == True:
        if(any(np.in1d(p1_rangeY,p2_rangeY))) == True:
#        Established that point 1 and 2 are the base.
            referenceY = (point1[0][1]+point2[0][1])/2
            if(point3[0][1] > referenceY):
                print("Down Arrow")
                orientation.append("2")
            else:
                print("Up Arrow")
                orientation.append("0")
        elif(any(np.in1d(p2_rangeY,p3_rangeY))) == True:
#        print("REE")
#        Established that point 2 and 3 are the base.
            referenceY = (point2[0][1]+point3[0][1])/2
            if(point1[0][1] > referenceY):
                print("Down Arrow")
                orientation.append("2")
            else:
                print("Up Arrow")
                orientation.append("0")
        else:
#        Established that point 2 and 3 are the base.
            referenceY = (point3[0][1]+point1[0][1])/2
            if(point2[0][1] > referenceY):
                print("Down Arrow")
                orientation.append("2")
            else:
                print("Up Arrow")
                orientation.append("0")

    else:
        print("Can't Detect Heading")
        
    return orientation
#def decodeMaze(horList,newVertList):
#    msg1 = []
#    msg2 = []
#    msg3 = []
#    msg4 = []
#    msg5 = []
#    msg6 = []
#    msg7 = []
#    msg8 = []
#    msg9 = []
#    msg10 = []
#    msg11 = []
#    
#    horString = "".join(horList)
#    vertString = "".join(newVertList)   
#    
#    for i in range(9):
#        if(horString[i] == "1"):
#            msg1.append("--- ")
#        else:
#            msg1.append("    ")
#    
#    for i in range(9, 18):
#        if(horString[i] == "1"):
#            msg3.append("--- ")
#        else:
#            msg3.append("    ")
#    
#    for i in range(18, 27):
#        if(horString[i] == "1"):
#            msg5.append("--- ")
#        else:
#            msg5.append("    ")    
# 
#    for i in range(27, 36):
#        if(horString[i] == "1"):
#            msg7.append("--- ")
#        else:
#            msg7.append("    ")
#
#    for i in range(36, 45):
#        if(horString[i] == "1"):
#            msg9.append("--- ")
#        else:
#            msg9.append("    ")
#
#    for i in range(45, 54):
#        if(horString[i] == "1"):
#            msg11.append("--- ")
#        else:
#            msg11.append("    ")            
#    
#    for i in range(10):
#        if(vertString[i] == "1"):
#            msg2.append("|   ")
#        else:
#            msg2.append("    ")
#            
#    for i in range(10,20):
#        if(vertString[i] == "1"):
#            msg4.append("|   ")
#        else:
#            msg4.append("    ")            
# 
#    for i in range(20,30):
#        if(vertString[i] == "1"):
#            msg6.append("|   ")
#        else:
#            msg6.append("    ")
#            
#    for i in range(30,40):
#        if(vertString[i] == "1"):
#            msg8.append("|   ")
#        else:
#            msg8.append("    ") 
#
#    for i in range(40,50):
#        if(vertString[i] == "1"):
#            msg10.append("|   ")
#        else:
#            msg10.append("    ")             
    
#    print(msg10)
#    msg1 = "".join(msg1)
#    msg2 = "".join(msg2)
#    msg3 = "".join(msg3)
#    msg4 = "".join(msg4)
#    msg5 = "".join(msg5)
#    msg6 = "".join(msg6)
#    msg7 = "".join(msg7)
#    msg8 = "".join(msg8)
#    msg9 = "".join(msg9)
#    msg10 = "".join(msg10)    
#    msg11 = "".join(msg11)    
#    
#    bluetooth = serial.Serial('COM5',9600)    
#    bluetooth.write(msg1.encode())
#    bluetooth.write("\n".encode())
#    time.sleep(1)
#    bluetooth.write(msg2.encode())
#    bluetooth.write("\n".encode())
#    time.sleep(1)
#    bluetooth.write(msg3.encode())
#    bluetooth.write("\n".encode())
#    time.sleep(1)
#    bluetooth.write(msg4.encode())
#    bluetooth.write("\n".encode())
#    time.sleep(1)
#    bluetooth.write(msg5.encode())
#    bluetooth.write("\n".encode())
#    time.sleep(1)
#    bluetooth.write(msg6.encode())
#    bluetooth.write("\n".encode())    
#    time.sleep(1)
#    bluetooth.write(msg7.encode())
#    bluetooth.write("\n".encode())    
#    time.sleep(1)
#    bluetooth.write(msg8.encode())
#    bluetooth.write("\n".encode())    
#    time.sleep(1) 
#    bluetooth.write(msg9.encode())
#    bluetooth.write("\n".encode())    
#    time.sleep(1)
#    bluetooth.write(msg10.encode())
#    bluetooth.write("\n".encode())    
#    time.sleep(1)
#    bluetooth.write(msg11.encode())
#    bluetooth.write("\n".encode())    
#    time.sleep(1)
#
#    
#    
#    print(msg1)
#    print(msg2)
#    print(msg3)
#    print(msg4)
#    print(msg5)
#    print(msg6)
#    print(msg7)
#    print(msg8)
#    print(msg9)
#    print(msg10)
#    print(msg11)
#    
#    return
    
def cropHorList(horList):
    
    croppedList = []
    for i in range(9,45):
        croppedList.append(horList[i])
    
#    print(croppedList)
    return croppedList

def cropVertList(newVertList):
    
    croppedList = []
#    for i in range(5,45):
#        croppedList.append(newVertList[i])
    for i in range(1,9):
        croppedList.append(newVertList[i])
    
    for i in range(11,19):
        croppedList.append(newVertList[i])    

    for i in range(21,29):
        croppedList.append(newVertList[i])
    
    for i in range(31,39):
        croppedList.append(newVertList[i])

    for i in range(41,49):
        croppedList.append(newVertList[i])        
        
    print(croppedList)
    return croppedList

def calandraCompatible(location,orientation,horList,vertList):
    
# =============================================================================
# Message string that will be sent to calandra's program.
# =============================================================================
    message = []  
#    bluetooth.write(msg1.encode())
#    bluetooth.write("\n".encode())
    
#   Appends the start of the message
    message.append("SS")
#   Starting Location
    message.append(location[0])
#   Direction
    message.append("D")
    if orientation == []:
        print("Fucked")
    else:
        message.append(orientation[0])
#   Horizontal Vector 
    message.append("H")
    for i in range(len(horList)):
        message.append(horList[i])
#   Vertical Vector 
    message.append("V")
    for i in range(len(vertList)):
        message.append(vertList[i])
#   End 
    message.append("E") 
    message = "".join(message)
    
    print(message)
#    port = 'COM4'
#    bluetooth = serial.Serial('COM4',115200,timeout = 3)  
#    print("Sending Message")
##    bluetooth.flushInput()
##    bluetooth.flush()
#    bluetooth.write(message.encode())
#    bluetooth.write('\n'.encode())
#    print("Message Sent")
    return

def liveTracking(vs):
    
# =============================================================================
# Vehicle Tracking Attempt 2
# =============================================================================
# construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
                    help="path to the (optional) video file")
    ap.add_argument("-b", "--buffer", type=int, default=1000000,
                    help="max buffer size")
    args = vars(ap.parse_args())
# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
    greenLower = (0, 0, 0)
    greenUpper = (180, 255, 5)
    pts = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab the reference
# to the webcam
    if not args.get("video", False):
#        print("Yeet")
        shit =VideoStream(src=0).start()
#        vs = VideoStream(src=1,resolution = (1920,1080),framerate = 30).start()
#        vs = cv2.VideoCapture(cv2.CAP_DSHOW)
#        vs.set(cv2.CAP_PROP_FPS, 30)
#        vs.set(cv2.CAP_PROP_FRAME_WIDTH,800)
#        vs.set(cv2.CAP_PROP_FRAME_HEIGHT,500)
#        vs.set(3,1920)
#        vs.set(4,1080)
# otherwise, grab a reference to the video file
    else:
        vs = cv2.VideoCapture(0)

# allow the camera or video file to warm up
#time.sleep(2.0)
# keep looping
    while True:
    # grab the current frame
        print("HERE")
        frame = vs.read()
    # handle the frame from VideoCapture or VideoStream
        frame = frame[1] if args.get("video", False) else frame

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
        if frame is None:
            break

    # resize the frame, blur it, and convert it to the HSV
    # color space
#        frame = imutils.resize(frame, width=800,height=500)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
#        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
        mask = cv2.inRange(hsv, greenLower, greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        # find contours in the mask and initialize the current
    # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

    # only proceed if at least one contour was found
        if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
            if radius > 40 and radius < 50:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
                print(radius)
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                pts.appendleft(center)
    # update the points queue
#        pts.appendleft(center)
        # loop over the set of tracked points
        for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
            if pts[i - 1] is None or pts[i] is None:
                continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
#        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
            cv2.line(frame,pts[i - 1], pts[i], (0, 0, 255), 5)

    # show the frame to our screen
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break

# if we are not using a video file, stop the camera video stream
    if not args.get("video", False):
        vs.stop()

# otherwise, release the camera
    else:
        vs.release()

# close all windows
    cv2.destroyAllWindows()

# =============================================================================
# Main Function        
# =============================================================================
snap = startCamera()
cam = startVideoStream()
#displayPicture(snap)
redMaze,horList,vertList = redOverLay(snap)
displayPicture(redMaze)
car,roi = findCar(snap)
orientation = getOrientation(roi)
#print(orientation)
#displayPicture(car) 
newVertList = reArrangeVertList(vertList)
croppedHorList = cropHorList(horList)
croppedVertList = cropVertList(newVertList)
calandraCompatible(car,orientation,croppedHorList,croppedVertList)
#decodeMaze(horList,newVertList)
#liveTracking(cam)


    