import cv2 as cv
import numpy as np
import module as m
import time

# Variables
COUNTER = 0
TOTAL_BLINKS = 0
CLOSED_EYES_FRAME = 3
cameraID = 0
videoPath = "Video/Your Eyes Independently_Trim5.mp4"
# variables for frame rate.
FRAME_COUNTER = 0
START_TIME = time.time()
FPS = 0


# creating camera object
camera = cv.VideoCapture(0)


# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
f = camera.get(cv.CAP_PROP_FPS)
width = camera.get(cv.CAP_PROP_FRAME_WIDTH)
height = camera.get(cv.CAP_PROP_FRAME_HEIGHT)
print(width, height, f)
fileName = videoPath.split('/')[1]
name = fileName.split('.')[0]
print(name)


while True:
    FRAME_COUNTER += 1
    ret, frame = camera.read()
    if ret == False:
        break

    # converting frame into Grey image.
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    height, width = grayFrame.shape
    circleCenter = (int(width/2), 50)
    # calling the face detector funciton
    image, face = m.faceDetector(frame, grayFrame)
    if face is not None:

        # calling landmarks detector funciton.
        image, PointList = m.faceLandmakDetector(frame, grayFrame, face, True)

        cv.putText(frame, f'FPS: {round(FPS,1)}',
                   (460, 20), m.fonts, 0.7, m.YELLOW, 2)
        RightEyePoint = PointList[36:42]
        LeftEyePoint = PointList[42:48]
        leftRatio, topMid, bottomMid = m.blinkDetector(LeftEyePoint)
        rightRatio, rTop, rBottom = m.blinkDetector(RightEyePoint)
        

        blinkRatio = (leftRatio + rightRatio)/2
       

        if blinkRatio > 4:
            COUNTER += 1
            
        else:
            if COUNTER > CLOSED_EYES_FRAME:
                TOTAL_BLINKS += 1
                COUNTER = 0
        
        for p in LeftEyePoint:
             cv.circle(image, p, 1, m.YELLOW, 1)

        for p in RightEyePoint:
             cv.circle(image, p, 1, m.YELLOW, 1)

        mask, pos, color, eyeImage, thresholdEye = m.EyeTracking(frame, grayFrame, RightEyePoint) # maske ve g�zbebe�i

        #mask, pos, color = m.EyeTracking(frame, grayFrame, RightEyePoint) # maske ve g�z�n g�zlemlenmedi�i hali
        maskleft, leftPos, leftColor , eyeImageLeft, thresholdEyeLeft= m.EyeTracking(frame, grayFrame, LeftEyePoint) # maske ve g�zbebe�i

        cv.imshow("mask",mask)
        cv.imshow("threshold",thresholdEye)
        cv.imshow('EyeImage',eyeImage)

        # draw background as line where we put text.
        cv.line(image, (30, 90), (100, 90), color[0], 30)
        cv.line(image, (25, 50), (135, 50), m.WHITE, 30)
        cv.line(image, (int(width-150), 50), (int(width-45), 50), m.WHITE, 30)
        cv.line(image, (int(width-140), 90),
                (int(width-60), 90), leftColor[0], 30)

        # writing text on above line
        cv.putText(image, f'{pos}', (35, 95), m.fonts, 0.6, color[1], 2)
        cv.putText(image, f'{leftPos}', (int(width-140), 95),
                   m.fonts, 0.6, leftColor[1], 2)
        cv.putText(image, f'Right Eye', (35, 55), m.fonts, 0.6, color[1], 2)
        cv.putText(image, f'Left Eye', (int(width-145), 55),
                   m.fonts, 0.6, leftColor[1], 2)

        # showing the frame on the screen
        cv.imshow('Frame', image)
    else:
        cv.imshow('Frame', frame)

    # Recoder.write(frame)
    # calculating the seconds
    SECONDS = time.time() - START_TIME
    # calculating the frame rate
    FPS = FRAME_COUNTER/SECONDS
    # print(FPS)
    # defining the key to Quit the Loop

    key = cv.waitKey(1)

    # if q is pressed on keyboard: quit
    if key == ord('q'):
        break
camera.release()
cv.destroyAllWindows()