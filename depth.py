
import cv2
import depthai as dai
import numpy as np
import os
import pathlib
import math
from PIL import Image
import sched, time
import datetime
# from depthai_sdk import OakCamera
from itertools import cycle
from scipy import ndimage

class inputManager():
    def __init__(self, cP = [], fP = [], l = 0, r = 0, t = 0, b = 0):
        self.polygonPoints = []
        self.validCutPoints = cP
        self.floorPoints = fP
        self.cutLeft = l
        self.cutRight = r
        self.cutTop = t
        self.cutBot = b
        self.lineColor = (1, 1, 1)
        self.lineColor4 = (0, 1, 0)
        self.lineColor5 = (1, 0, 0)
        self.lineColor3 = (.2, .2, .2)
        self.lineColor2 = (50, 50, 50)
        self.pointColor = (255, 255, 255)
    
    def mouseInput(self, event, x, y, buttons, user_param):
        # global mouseX, mouseY
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.polygonPoints.append((x,y))
        elif event == cv2.EVENT_RBUTTONDBLCLK:
            self.polygonPoints = []
    
    def editCutLeft(self, newVal):
        self.cutLeft = newVal

    def editCutRight(self, newVal):
        self.cutRight = newVal

    def editCutTop(self, newVal):
        self.cutTop = newVal

    def editCutBot(self, newVal):
        self.cutBot = newVal

    def setPolyAsFloor(self):
        self.floorPoints = self.polygonPoints
    
    def setPolyAsValid(self):
        self.validCutPoints = self.polygonPoints
    
    def editFrame(self, frame):
        frameBackupA = frame
        frameBackupB = frame
        frameBackup = frame
        frameBackground = frame
        frameBackgroundA = frame
        frameBackgroundB = frame
        # canvas = np.zeros((800,1280,4),dtype='uint8')
        if (self.validCutPoints != []):
            canvas = np.zeros((800,1280,3),dtype='uint8')
            cv2.fillPoly(canvas, np.array([self.validCutPoints]), self.lineColor)
            # canvas[canvas == 0] = self.lineColor5
            # cut out
            frameBackupA = (frameBackupA * (canvas))
            # opposite
            framecut1 = np.zeros_like(frame)
            framecut1 = cv2.subtract(frameBackup, frameBackupA,framecut1)
            # color opposite
            # framecut1 = framecut1 * self.lineColor4
            framecut1 = cv2.addWeighted(framecut1, 0.5,framecut1, 0 ,0)
            
            frame = cv2.add(frameBackupA,framecut1, frame)

        if (self.floorPoints != []):
            canvas = np.zeros((800,1280,3),dtype='uint8')
            cv2.fillPoly(canvas, np.array([self.floorPoints]), self.lineColor)
            # canvas[canvas == 0] = self.lineColor4
            # cut out
            frameBackupB = frameBackupB * (canvas)

            framecut2 = np.zeros_like(frame)
            framecut2 = cv2.subtract(frameBackup, frameBackupB,framecut2)
            framecut2 = cv2.addWeighted(framecut2, 0.5,framecut2, 0 ,0)

            frame = cv2.add(frameBackupB,framecut2, frame)

        if (self.polygonPoints != []):
            canvas = np.zeros((800,1280,3),dtype='uint8')
            cv2.polylines(canvas, np.array([self.polygonPoints]), True, self.lineColor2, 2)
            frame = frame + canvas
            for point in self.polygonPoints:
                cv2.circle(frame, point, 4, self.pointColor, 2)
        # TODO cut the sides of the frame
        if (self.cutTop != 0):
            # Draw a diagonal blue line with thickness of 5 px
            cv2.line(frame,(0,0),(1280,0),(0,0,0),self.cutTop*2)
            # for i in range(self.cutTop):
            #     cv2.line(frame,(i,0),(i,800),(255,0,0),4)
        if (self.cutBot != 0):
            cv2.line(frame,(0,800),(1280,800),(0,0,0),self.cutBot*2)
        if (self.cutLeft != 0):
            cv2.line(frame,(0,0),(0,800),(0,0,0),self.cutLeft*2)
        if (self.cutRight != 0):
            cv2.line(frame,(1280,0),(1280,800),(0,0,0),self.cutRight*2)
            

        return frame
    
    def writeConfig(self, path, fileCounter):
        configname = "depth" + str(fileCounter) + ".txt"
        thisConfigPath = os.path.join(path, configname)
        print(str(thisConfigPath))
        file = open(thisConfigPath, 'w')
        file.write(str(self.validCutPoints))
        file.write('\n')
        file.write(str(self.floorPoints))
        file.write('\n')
        file.write(str(self.cutLeft))
        file.write('\n')
        file.write(str(self.cutRight))
        file.write('\n')
        file.write(str(self.cutBot))
        file.write('\n')
        file.write(str(self.cutTop))
        return



def printDiagram(array = []):
    if array:
        s = int(np.sqrt(array.size))

# dont use it - wip
def myMeanFilter(i0 ,i1, i2,i3,i4,i5,i6,i7,i8):
    # idea: if the 8-neighborhood has >= 6 nearly same pixelvalues take them, else give back the original pixel
    # make a list with every pixel +-1 --> 8*3 entries
    # i0 is the main pixel
    pixelValueList = []
    pixelValueList.extend((i1-1,i1,i1+1,i2-1,i2,i2+1,i3-1,i3,i3+1,i4-1,i4,i4+1,i5-1,i5,i5+1,i6-1,i6,i6+1,i7-1,i7,i7+1,i8-1,i8,i8+1))

    #sorted(set([i for i in mylist if mylist.count(i)>2]))
    pixelValueList = sorted(set([i for i in pixelValueList if pixelValueList.count(i)>5]))
    if pixelValueList == []:
        return i0
    else:
        return pixelValueList[0]+1

# dont use it - wip
def mean(i0 ,i1, i2,i3,i4,i5,i6,i7,i8):
    pixelValueList = []
    pixelValueList.extend((i1-1,i1,i1+1,i2-1,i2,i2+1,i3-1,i3,i3+1,i4-1,i4,i4+1,i5-1,i5,i5+1,i6-1,i6,i6+1,i7-1,i7,i7+1,i8-1,i8,i8+1))

    pixelValueList.sort()
    return pixelValueList[11]      

def printImg(imagePath, frame2, fileCounter, frame3 = None, filter = True):
    image_name = "depth" + str(fileCounter) + ".png"
    rgb_name = "rgb" + str(fileCounter) + ".png"
    thisImagePath = os.path.join(imagePath, image_name)
    parentFolder = pathlib.Path(imagePath).parent
    # TODO path
    parentFolder = os.path.join(parentFolder, "outputRGB\\")
    thisRGBPath = os.path.join(parentFolder, rgb_name)
    print(thisImagePath)
    
    newFrame = ndimage.median_filter(frame2, size=13)
    thisImagePath2 = os.path.join(imagePath, "f13"+image_name)
    cv2.imwrite(thisImagePath2, newFrame)
    
    if frame3.any():
        print(thisRGBPath)
        cv2.imwrite(thisRGBPath, frame3)
    
def printImgPeriodical(scheduler, interval, action, actionargs=()):
    scheduler.enter(interval, 1, printImgPeriodical,(scheduler, interval, action, actionargs))
    action(*actionargs)

def removeOldFiles(imagePath):
    oldFilePaths = list (pathlib.Path(imagePath).glob('*.png'))
    oldConfFilePaths = list (pathlib.Path(imagePath).glob('*.txt'))
    print("len OLD png files: " + str(len(oldFilePaths)))
    for file in oldFilePaths:
        os.remove(file) 
    for file in oldConfFilePaths:
        os.remove(file) 

def calcVolume(path, changeOfPerspektive=False, cutSides =True):
    volume = 0
    pathList = list(pathlib.Path(path).glob('*.png'))
    pathListImgConfig = list(pathlib.Path(path).glob('*.txt'))
    numImage = len(pathList)
    depthImage = np.zeros((800,1280),np.int32)
    measurementImage = np.zeros((800,1280),np.int8)
    counter = 0

    valid = None
    floor = None
    cutLeft = 0
    cutRight = 0
    cutBot = 0
    cutTop = 0
    for imagePath in pathList:
        configFile = open(pathListImgConfig[counter],'r')
        # image --> median & something more maybe
        thisImage = np.array(Image.open(imagePath))
        depthImage = np.add(depthImage , thisImage)
        valid = configFile.readline()
        floor = configFile.readline()
        cutLeft = int (configFile.readline())
        cutRight = int (configFile.readline())
        cutBot = int (configFile.readline())
        cutTop = int (configFile.readline())
        print("valid: " + str(valid))
        print("floor: " + str(floor))
        print("cut: " + str(cutLeft) + ", " + str(cutRight) + ", " + str(cutBot) + ", " + str(cutTop))
        # maybe new np.array with a counter for all !=0 elements (thresholding) --> divide by this later (think about 0)
        # measurementImage = np.add(measurementImage,np.where(thisImage > 1, 1, 0))
        measurementImage = np.add(measurementImage,np.where(thisImage != 0, 1, 0))
        counter += 1
    #  image should now be in mm
    # image now float64
    # depthImage = depthImage/(numImage)

    printData = False    
    if printData:
        for j in range(80):
            file = open('outputData/data'+str(j)+'.txt', 'w')
            for i in range(1279):
                file.writelines(str(frame2[j*10,i]) + "\n")

    # TODO use measurementImage properly
    max = np.max(measurementImage)
    depthImage = np.where(measurementImage >= max/3, depthImage/measurementImage, 0)

    # Got this from: https://discuss.luxonis.com/d/339-naive-question-regarding-stereodepth-disparity-and-depth-outputs/7
    # ‘W’ is the width of an image in pixels, which is constant for a given resolution. 
    # ‘F’ is the width of an image in cm at a distance ‘D’ cm from the cameras; 
    # F varies with D.

    # F = 2 * D * tan(HFOV/2) [cm]
        # F = D * 2 * tan(HFOV/2)
    # F^2 = 2 * D * tan(77/2)	[cm²]
        # F^2 = F * F = areaImage
    # Volume = D * F^2	[cm³]

    # Camera Specs		    Color camera			Stereo pair
    # Sensor			    IMX378 (PY004 AF, PY052 FF)	OV9282 (PY003)
    # DFOV / HFOV / VFOV	81° / 69° / 55°			82° / 77° / 53°

    # depthEst = 1000 #mm
    # imageWidth = 2 * depthEst * math.tan((77 * (math.pi/180))/2) #mm
    # imageHeigth = 2 * depthEst * math.tan((53 * (math.pi/180))/2) #mm

    # for testing purposes lets use a other depthImage value
    # depthImage = np.full((800,1280),2000,np.int32)

    hFov = 77
    vFov = 53
    oak_d_pro = True
    # TODO pro values
    if oak_d_pro:
        hFov = 80
        vFov = 55
    
    imageWidth = 2 * depthImage * math.tan(hFov/2 * (math.pi/180)) #mm
    imageHeigth = 2 * depthImage * math.tan(vFov/2 * (math.pi/180)) #mm
    
    imageVolDim = np.zeros((800,1280),np.float64) #mm

    for i in range(800):
        for j in range(1280): 
            imageVolDim[i,j] = (2 * depthImage[i,j] * (math.tan(hFov/1280 * (math.pi/180)) + (math.tan(vFov/800 * (math.pi/180))))) * depthImage[i,j]

    imageArea = imageWidth * imageHeigth #mm²

    # create a function or np.array for all the areas of the pixels
    areaImage = np.full((800,1280),imageArea/(800*1280))
    
    # volume np array
    volume_image = np.multiply(depthImage,areaImage)

    # TODO use valid and floor for cutting
    newVolumeImage = volume_image
    if valid is not None:
        # need to multiply it with the mask which can be access by the points
        newVolumeImage = newVolumeImage 

    if floor is not None:
        newVolumeImage = newVolumeImage
    
    
    # cut the image -10 bottom, -25 top, -40 right, -60 left
    if cutSides:
        # (800,1280)
        newVolumeImage = newVolumeImage[(cutBot):(799-cutTop),(cutLeft):(1279-cutRight)]
        # pil_img2 = Image.fromarray(newVolumeImage).convert('RGB')
        # pil_img2.save(os.path.join(pathlib.Path(path).parent, "imageOut2.png"))
        volume_image = newVolumeImage

        # newVolumeImage2 = imageVolDim[10:775,61:1255]

    pil_img2 = Image.fromarray(newVolumeImage).convert('RGB')
    pil_img2.save(os.path.join(pathlib.Path(path).parent, "imageOut3.png"))

    volume = np.sum(volume_image, dtype=np.float64)
    volume = volume*0.000000001
    print("volume = " + str(volume))
    # volume2 = np.sum(newVolumeImage2, dtype=np.float64)
    # volume2 = volume2*0.000000001
    # print("volume2 = " + str(volume2))
    return depthImage, volume


def printNice(path, depthImage):
    image_nice = np.zeros((800,1280,3),dtype='uint8')
    
    print(image_nice[400,640])

    # would be better to use:
    # wavelengthImage = depthImage/50 + 400
    # but this is more visualizing
    wavelengthImage = depthImage/15 + 400
    
    print(wavelengthImage[400,640])

    # how to calculate the colors: https://stackoverflow.com/questions/3407942/rgb-values-of-visible-spectrum/22681410#22681410
    for i in range(800):
        for j in range(1280):
            l = wavelengthImage[i,j]
            if ((l>=400.0)and(l<410.0)):
                t=(l-400.0)/(410.0-400.0)
                image_nice[i,j,0] =((0.33*t)-(0.20*t*t))*255
            elif ((l>=410.0)and(l<475.0)):
                t=(l-410.0)/(475.0-410.0)
                image_nice[i,j,0] =(0.14         -(0.13*t*t))*255
            elif ((l>=545.0)and(l<595.0)):
                t=(l-545.0)/(595.0-545.0)
                image_nice[i,j,0] =   ( (1.98*t)-(     t*t))*255
            elif ((l>=595.0)and(l<650.0)):
                t=(l-595.0)/(650.0-595.0)
                image_nice[i,j,0] =(0.98+(0.06*t)-(0.40*t*t))*255
            elif ((l>=650.0)and(l<700.0)):
                t=(l-650.0)/(700.0-650.0)
                image_nice[i,j,0] =(0.65-(0.84*t)+(0.20*t*t))*255
            if ((l>=415.0)and(l<475.0)):
                t=(l-415.0)/(475.0-415.0)
                image_nice[i,j,1]=((0.80*t*t))*255
            elif ((l>=475.0)and(l<590.0)):
                t=(l-475.0)/(590.0-475.0)
                image_nice[i,j,1]=(0.8 +(0.76*t)-(0.80*t*t))*255
            elif ((l>=585.0)and(l<639.0)):
                t=(l-585.0)/(639.0-585.0)
                image_nice[i,j,1]=(0.84-(0.84*t) )*255
            if ((l>=400.0)and(l<475.0)):
                t=(l-400.0)/(475.0-400.0)
                image_nice[i,j,2]=((2.20*t)-(1.50*t*t))*255
            elif ((l>=475.0)and(l<560.0)):
                t=(l-475.0)/(560.0-475.0)
                image_nice[i,j,2]=(0.7 -(t)+(0.30*t*t)   )*255
            
    pil_img = Image.fromarray(image_nice)
    pil_img.save(os.path.join(pathlib.Path(path).parent, "imageOut.png"))



filePath = os.path.__file__ 
ROOT_DIR = pathlib.Path(__file__).parent
# print(ROOT_DIR)
# TODO append folder in a different way
imagePath = os.path.join(ROOT_DIR, 'output\\')
# print(imagePath)

# TODO append folder in a different way
rgbImagePath = os.path.join(ROOT_DIR, 'outputRGB\\')
removeOldFiles(imagePath)
removeOldFiles(rgbImagePath)

# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
extended_disparity = True
# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = True
# Better handling for occlusions:
lr_check = True

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs

depth = pipeline.create(dai.node.StereoDepth)

monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("disparity")
xout2 = pipeline.create(dai.node.XLinkOut)
xout2.setStreamName("depth")

# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# colorcamera
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)

#1
ispOut = pipeline.create(dai.node.XLinkOut)
videoOut = pipeline.create(dai.node.XLinkOut)

#1
ispOut.setStreamName('isp')
videoOut.setStreamName('video')


# Linking
#1
camRgb.isp.link(ispOut.input)
camRgb.video.link(videoOut.input)

# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
depth.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)

depth.setLeftRightCheck(lr_check)
depth.setExtendedDisparity(extended_disparity)
depth.setSubpixel(subpixel)
depth.setDepthAlign(align=dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.CENTER)

config = depth.initialConfig.get()

config.postProcessing.speckleFilter.enable = False
config.postProcessing.speckleFilter.speckleRange = 50
config.postProcessing.temporalFilter.enable = True
config.postProcessing.spatialFilter.enable = True
config.postProcessing.spatialFilter.holeFillingRadius = 2
config.postProcessing.spatialFilter.numIterations = 1
config.postProcessing.thresholdFilter.minRange = 400
config.postProcessing.thresholdFilter.maxRange = 15000
config.postProcessing.decimationFilter.decimationFactor = 1
depth.initialConfig.set(config)

# Linking
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)
depth.disparity.link(xout.input)
depth.depth.link(xout2.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # activate laser
    device.setIrLaserDotProjectorBrightness(800)
    device.setIrFloodLightBrightness(200)
    
    
    ispQueue = device.getOutputQueue('isp')
    videoQueue = device.getOutputQueue('video')
   
    # Defaults and limits for manual focus/exposure controls
    lensPos = 150
    expTime = 20000
    sensIso = 800    
    wbManual = 4000
    ae_comp = 0
    ae_lock = False
    awb_lock = False
    saturation = 0
    contrast = 0
    brightness = 0
    sharpness = 0
    luma_denoise = 0
    chroma_denoise = 0
    control = 'none'
    # show = False

    awb_mode = cycle([item for name, item in vars(dai.CameraControl.AutoWhiteBalanceMode).items() if name.isupper()])
    anti_banding_mode = cycle([item for name, item in vars(dai.CameraControl.AntiBandingMode).items() if name.isupper()])
    effect_mode = cycle([item for name, item in vars(dai.CameraControl.EffectMode).items() if name.isupper()])

    # Output queue will be used to get the disparity frames from the outputs defined above
    q = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
    q2 = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
   
    fileCounter = 0

    lastPrint = datetime.datetime.now()
    imgToPrint = 0
    intervall = 2
    baseVolume = 0

    iM = inputManager() 

    cv2.namedWindow('frameToEdit')

    cv2.setMouseCallback('frameToEdit', iM.mouseInput)

    cv2.namedWindow('test')

    cv2.createTrackbar('cut left', 'test', 0, 640-1, iM.editCutLeft)
    cv2.createTrackbar('cut right', 'test', 0, 640-1, iM.editCutRight)
    cv2.createTrackbar('cut top', 'test', 0, 400-1, iM.editCutTop)
    cv2.createTrackbar('cut bot', 'test', 0, 400-1, iM.editCutBot)

    showRgb = False
    showDepth = False
    showDisparityColor = False
    showRgbIsp = False
    
    while True:

        inDisparity = q.get()  # blocking call, will wait until a new data has arrived
        frame = inDisparity.getFrame()
        inDepth = q2.get()
        frame2 = inDepth.getCvFrame()

        if(True):
            vidFrames = videoQueue.tryGetAll()
            for vidFrame in vidFrames:
                frame3 = vidFrame.getCvFrame()
                down_width = 384*4
                down_height = 216*4
                down_points = (down_width, down_height)
                # print("size frame 3: " + str(frame3.shape))
                frame3_small = cv2.resize(frame3, down_points, interpolation= cv2.INTER_LINEAR)
                # print("size frame 3 small: " + str(frame3_small.shape))
                if showRgb:
                    cv2.imshow("rgb_vid",frame3_small)
            ispFrames = ispQueue.tryGetAll()
            for ispFrame in ispFrames:
                frame4 = ispFrame.getCvFrame()
                down_width = 405*3
                down_height = 304*3
                down_points = (down_width, down_height)
                # print("size frame 4: " + str(frame4.shape))
                frame4_small = cv2.resize(frame4, down_points, interpolation= cv2.INTER_LINEAR)
                # print("size frame 4 small: " + str(frame4_small.shape))
                if showRgbIsp:
                    cv2.imshow("rgb_isp",frame4_small)

        frame = (frame * (455 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)

        consistencyCheck = False
        if consistencyCheck:
            timeDifference = datetime.datetime.now() - lastPrint
            if timeDifference.seconds > 1:
                consistencyFile = open('consistency1.txt', 'a')
                consistencyFile.writelines(str(frame2[200,320])+"\n")
                consistencyFile.close()
                consistencyFile = open('consistency2.txt', 'a')
                consistencyFile.writelines(str(frame2[200,960])+"\n")
                consistencyFile.close()
                consistencyFile = open('consistency3.txt', 'a')
                consistencyFile.writelines(str(frame2[400,640])+"\n")
                consistencyFile.close()
                consistencyFile = open('consistency4.txt', 'a')
                consistencyFile.writelines(str(frame2[600,320])+"\n")
                consistencyFile.close()
                consistencyFile = open('consistency5.txt', 'a')
                consistencyFile.writelines(str(frame2[600,960])+"\n")
                consistencyFile.close()

        # print("400,640: " + str(frame2[400,640]))
        if showDepth:
            cv2.imshow("depth", frame2)
        # print(str(frame2[300,300]))
        # print(str(frame2.shape))
        # print(str(frame2.size))
        # print(str(frame2.dtype ))
        # frame2 = cv2.medianBlur(frame2,7,frame2)
        # cv2.imshow("depth_median", frame2)
        
        # Available color maps: https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html
        frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
    
        if showDisparityColor:
            cv2.imshow("disparity_color", frame)
        
        cv2.imshow('frameToEdit',iM.editFrame(frame))

        # cut the image -10 bottom, -25 top, -40 right, -60 left
        # frame5 = frame[cutBot:((800-cutBot)-cutTop),cutLeft:((1280-cutLeft)-cutRight)]
        # cv2.imshow("disparity_color", frame5)
        
        if imgToPrint > 0:
            timeDifference = datetime.datetime.now() - lastPrint
            if timeDifference.seconds > 2:
                printImg(imagePath, frame2, fileCounter, frame4)
                iM.writeConfig(imagePath, fileCounter)
                lastPrint = datetime.datetime.now()
                fileCounter += 1
                imgToPrint -= 1

        debug = False
        if debug:
            print("poly: " + str(iM.polygonPoints))  
            print("cut: " + str(iM.cutLeft))  

        pressedkey = cv2.waitKey(50) & 0xFF
        if pressedkey == ord('q'):
            break
        elif pressedkey == ord('w'):
            # image_name = "depth" + str(fileCounter) + ".png"
            # thisImagePath = os.path.join(imagePath, image_name)
            # # print(thisImagePath)
            # cv2.imwrite(thisImagePath, frame2)
            printImg(imagePath, frame2, fileCounter, frame4)
            iM.writeConfig(imagePath, fileCounter)
            fileCounter += 1
        elif pressedkey == ord('r'):
            print(calcVolume(imagePath)[1])
        elif pressedkey == ord('t'):
            imgToPrint += 10
            # myScheduler = sched.scheduler(time.time,time.sleep)
            # myScheduler = sched.scheduler()
            # for i in range(n):
            #     myScheduler.enter(i*intervall,1,printImg, (imagePath, frame2, fileCounter))
            #     fileCounter += 1
            # myScheduler.run()
        elif pressedkey == ord('e'):
            printNice(imagePath, calcVolume(imagePath)[0])
        elif pressedkey == ord('z'):
            print("baseVolume before: "+ str(baseVolume))
            baseVolume = calcVolume(imagePath)[1]
            removeOldFiles(imagePath)
            print("baseVolume after: "+  str(baseVolume))
        elif pressedkey == ord('u'):
            removeOldFiles(imagePath)
        elif pressedkey == ord('i'):
            print("baseVolume: "+ str(baseVolume))
            thisVolume  = calcVolume(imagePath)[1]
            print("this Volume: " + str(thisVolume))
            volumeDifference = baseVolume - thisVolume
            print("volume difference: " + str(volumeDifference))
        elif pressedkey == ord('o'):
            printDiagram()
        elif pressedkey == ord('f'):
            iM.setPolyAsFloor()
        elif pressedkey == ord('v'):
            iM.setPolyAsValid()
        elif pressedkey == ord('p'):
            # 1280x800
            for i in range(1279):
                print(str(i) + "   \t" + str(frame2[400,i]))
            
            print("50   " + str(frame2[400,50]))
            print("100  " + str(frame2[400,150]))
            print("150  " + str(frame2[400,200]))
            print("200  " + str(frame2[400,250]))
            print("250  " + str(frame2[400,300]))
            print("300  " + str(frame2[400,350]))
            print("350  " + str(frame2[400,400]))
            print("400  " + str(frame2[400,450]))
            print("450  " + str(frame2[400,500]))
            print("500  " + str(frame2[400,550]))
            print("550  " + str(frame2[400,600]))
            print("600  " + str(frame2[400,650]))
            print("650  " + str(frame2[400,700]))
            print("700  " + str(frame2[400,750]))
            print("750  " + str(frame2[400,800]))
            print("800  " + str(frame2[400,850]))
            print("850  " + str(frame2[400,900]))
            print("900  " + str(frame2[400,950]))
            print("950  " + str(frame2[400,1000]))
            print("1000 " + str(frame2[400,1050]))
            print("1050 " + str(frame2[400,1100]))
            print("1100 " + str(frame2[400,1150]))
            print("1150 " + str(frame2[400,1200]))
            print("1200 " + str(frame2[400,1250]))


# Ablauf
# q: quit
# w: write 1 image
# r: calc volume
# t: write 10 img
# e: print nice
# z: set base volume
# u: delete images
# i: calc difference to base volume
# p: print pixel value at some positions
# f: FloorMode: double-click on the image to mark the floor area
# v: ValidMode: double-click on the image to mark the valid area
#
# 
