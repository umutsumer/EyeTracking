# coding=utf8
import sys
from turtle import setposition
import cv2 as cv
import numpy as np
import module as m
import time
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

class MainWindow(QWidget):
    
    def __init__(self):
        super(MainWindow, self).__init__()
        pozisyon = ""
        self.VBL = QVBoxLayout()

        self.FeedLabel = QLabel()
        self.VBL.addWidget(self.FeedLabel)
        
        self.CancelBTN = QPushButton("Cancel")
        self.CancelBTN.clicked.connect(self.CancelFeed)
        self.VBL.addWidget(self.CancelBTN)
        self.TextLabel = QLabel(pozisyon)
        self.VBL.addWidget(self.TextLabel)
        self.Worker1 = Worker1()

        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.setLayout(self.VBL)
    def changeText(self, str):
        self.TextLabel.setText(str)

    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def CancelFeed(self):
        self.Worker1.stop()

class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    def run(self):
        self.ThreadActive = True
        camera = cv.VideoCapture(0)
        while self.ThreadActive:
            frame, ret,  = Baslat.Camera(camera) # TODO
            #ret, frame = Capture.read()
            if ret:
                Image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                #FlippedImage = cv.flip(Image, 1)
                ConvertToQtFormat = QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
                #MainWindow.changeText(pos)
    def stop(self):
        self.ThreadActive = False
       
        cv.destroyAllWindows()
        self.quit()


class Baslat:
    
    def Camera(camera):
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

                #cv.putText(frame, f'FPS: {round(FPS,1)}',
                #           (460, 20), m.fonts, 0.7, m.YELLOW, 2)
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
        
                #for p in LeftEyePoint:   # SARI NOKTALAR
                     #cv.circle(image, p, 1, m.YELLOW, 1)

                #for p in RightEyePoint:                    # SARI NOKTALAR
                     #cv.circle(image, p, 1, m.YELLOW, 1)

                mask, pos, color, eyeImage, thresholdEye = m.EyeTracking(frame, grayFrame, RightEyePoint) # maske ve gözbebeği
                #mask, pos, color = m.EyeTracking(frame, grayFrame, RightEyePoint) # maske ve gözün gözlemlenmediği hali
                maskleft, leftPos, leftColor , eyeImageLeft, thresholdEyeLeft= m.EyeTracking(frame, grayFrame, LeftEyePoint) # maske ve gözbebeği

                #cv.imshow("mask",mask)
                #cv.imshow("threshold",thresholdEye)
                #cv.imshow('EyeImage',eyeImage)

                # draw background as line where we put text.
                #cv.line(image, (30, 90), (100, 90), color[0], 30)
                #cv.line(image, (25, 50), (135, 50), m.WHITE, 30)
                #cv.line(image, (int(width-150), 50), (int(width-45), 50), m.WHITE, 30)
                #cv.line(image, (int(width-140), 90),
                #        (int(width-60), 90), leftColor[0], 30)
                image = cv.flip(image, 1)

                # writing text on above line
                cv.putText(image, f'{pos}', (25, 95), m.fonts, 0.8, color[1], 2)
                #cv.putText(image, f'{leftPos}', (int(width-140), 95),
                #           m.fonts, 0.6, leftColor[1], 2)
                #cv.putText(image, f'Right Eye', (35, 55), m.fonts, 0.6, color[1], 2)
                #cv.putText(image, f'Left Eye', (int(width-145), 55),
                #           m.fonts, 0.6, leftColor[1], 2)

                # showing the frame on the screen
                #cv.imshow('Frame', image)
                return image, ret
                
            else:
                frame = cv.flip(frame, 1)
                #cv.imshow('Frame', frame)
                return frame, ret
                

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

