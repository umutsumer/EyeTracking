# coding=utf8
from asyncio.windows_events import NULL
from distutils.command.check import check
from email.errors import FirstHeaderLineIsContinuationDefect
from sqlite3 import Time
import sys
from tkinter import CENTER
from turtle import setposition
import cv2 as cv
import numpy as np
import module as m
import time
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import tensorflow as tf 
import mediapipe as mp
import math

flag = 0
blink = 0
i = 0
first_blink = False
check_second_blink = False
first_blink_time = 0
operation = ""
class MainWindow(QWidget):
    
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Cemal")
        self.VBL = QVBoxLayout()

        self.FeedLabel = QLabel()
        self.VBL.addWidget(self.FeedLabel)
        
        
        self.TextLabel = QLabel("")
        self.TextLabel.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(20)  # Set the font size to 20 points
        font.setBold(True)  # Set the font to be bold
        self.TextLabel.setFont(font)  # Apply the font to the text label
        self.TextLabel.setStyleSheet("color: black;")

        self.VBL.addWidget(self.TextLabel)


        self.TextLabel2 = QLabel("")
        self.TextLabel2.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(20)  # Set the font size to 20 points
        font.setBold(True)  # Set the font to be bold
        self.TextLabel2.setFont(font)  # Apply the font to the text label
        self.TextLabel2.setStyleSheet("color: red;")
        
        self.VBL.addWidget(self.TextLabel2)

        
        
        self.CancelBTN = QPushButton("Close")
        self.CancelBTN.clicked.connect(self.CancelFeed)
        self.VBL.addWidget(self.CancelBTN)
        self.Worker1 = Worker1()

        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.Worker1.PositionTextUpdate.connect(self.TextUpdateSlot)
        self.Worker1.OperationTextUpdate.connect(self.OpTextUpdateSlot)
        self.setLayout(self.VBL)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.clearText)
    def changeText(self, str):
        self.TextLabel.setText(str)

    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def TextUpdateSlot(self,text):
        self.TextLabel.setText(text.upper())

    def OpTextUpdateSlot(self,text):
        if text != "":
            self.TextLabel2.setText(text)
            self.timer.start(15000)
        

    def clearText(self):
        self.TextLabel2.clear()
        self.timer.stop()

    def CancelFeed(self):
        self.Worker1.stop()
        MainWindow.destroy(self)
        

class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    PositionTextUpdate = pyqtSignal(str)
    OperationTextUpdate = pyqtSignal(str)
    def run(self):
        self.ThreadActive = True
        cap = cv.VideoCapture(0)
        

        while self.ThreadActive:
            operation, pos, ret , frame  = Baslat.Camera(cap) # TODO
            
            
            if ret:
                Image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                ConvertToQtFormat = QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
                self.PositionTextUpdate.emit(pos)
                self.OperationTextUpdate.emit(operation)
    def stop(self):
        self.ThreadActive = False
        
       
        cv.destroyAllWindows()
        self.quit()


class Baslat:
    def Camera(cap):
        mp_face_mesh = mp.solutions.face_mesh
        LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        RIGHT_IRIS = [474, 475, 476, 477]
        LEFT_IRIS = [469, 470, 471, 472]
        L_H_LEFT = [33]
        L_H_RIGHT = [133]
        L_H_UP = [159]
        L_H_DOWN = [145]
        R_H_LEFT = [362]
        R_H_RIGHT = [263]
        R_H_UP = [386]
        R_H_DOWN = [374]

        def euclidean_distance(point1, point2):
            x1, y1 = point1.ravel()
            x2, y2 = point2.ravel()
            distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            return distance

        def iris_position(iris_center, right_point, left_point, up_point, down_point):
            center_to_right_dist = euclidean_distance(iris_center, right_point)
            lr_total_distance = euclidean_distance(right_point, left_point)
            lr_ratio = center_to_right_dist / lr_total_distance
    
            center_to_up_dist = euclidean_distance(iris_center, up_point)
            ud_total_distance = euclidean_distance(up_point, down_point)
            try:
                ud_ratio = center_to_up_dist / ud_total_distance
            except ZeroDivisionError:
                ud_ratio = 0

            iris_position = ""
            command = ""
            if ud_ratio > 0.42 and ud_ratio <= 0.58 and lr_ratio > 0.42 and lr_ratio <= 0.57:
                iris_position = "center"
                command = "STAYING STILL"
            elif ud_ratio >= 0.58 and lr_ratio > 0.42 and lr_ratio <= 0.57:
                iris_position = "down"
                command = "STOPPING"
            elif ud_ratio <= 0.42 and lr_ratio > 0.42 and lr_ratio <= 0.57:
                iris_position = "up"    
                command = "MOVING FORWARD"
            elif lr_ratio <= 0.42:
                iris_position = "right"
                command = "TURNING RIGHT"
            elif lr_ratio > 0.57:
                iris_position = "left"
                command = "TURNING LEFT"

            return command, iris_position, lr_ratio, ud_ratio


        global flag 
        global i 
        global first_blink 
        global check_second_blink 
        global first_blink_time
        global blink                    
        global operation

        with mp_face_mesh.FaceMesh(
            max_num_faces=1, 
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv.flip(frame,1)
                rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                img_h, img_w = frame.shape[:2]
                results = face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
                    (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
                    (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                    center_left = np.array([l_cx, l_cy], dtype = np.int32)
                    center_right = np.array([r_cx, r_cy], dtype = np.int32)
                    blink = 0
                    operation = ""
                    command, iris_pos, lr_ratio, ud_ratio = iris_position(center_right, mesh_points[R_H_RIGHT][0], mesh_points[R_H_LEFT][0], mesh_points[R_H_UP][0], mesh_points[R_H_DOWN][0])
                    total_distance = euclidean_distance(mesh_points[R_H_UP][0], mesh_points[R_H_DOWN][0])
                    if total_distance <= 3 and flag == 0:
                        # Only increment the blink count if this is the first blink in a sequence
                        flag = 1
                        i = i + 1
                        print("blink count:",i," command:",iris_pos)
                        if first_blink == False:
                            first_blink = True
                            first_blink_time = time.time()
                        elif check_second_blink:
                            # This is the second blink in a sequence, so reset the flags
                            first_blink = False
                            check_second_blink = False
                            operation = str(command)
                            print("Two blink detected ",operation)
                            blink = 1
                            
                    elif total_distance >3:
                        # Reset the flag if the user has fully opened their eyes
                        flag = 0
                        if first_blink:
                            check_second_blink = True

                    if first_blink and time.time() - first_blink_time > 5:
                        # If the user has blinked at least once and it has been more than 5 seconds, reset the flags
                        first_blink = False
                        check_second_blink = False
                        print("Too old.")

                return operation, iris_pos,ret, frame
                key = cv.waitKey(1)
                if key == ord('q'):
                    break

        cap.release()
        cv.destroyAllWindows()


