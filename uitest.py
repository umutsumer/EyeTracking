# coding=utf8
from asyncio.windows_events import NULL
from distutils.command.check import check
from email.errors import FirstHeaderLineIsContinuationDefect
from sqlite3 import Time
import sys
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
        cap = cv.VideoCapture(0)
        while self.ThreadActive:
            ret , frame  = Baslat.Camera(cap) # TODO
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
            ud_ratio = center_to_up_dist / ud_total_distance

            iris_position = ""

            if ud_ratio > 0.42 and ud_ratio <= 0.58 and lr_ratio > 0.42 and lr_ratio <= 0.57:
                iris_position = "center"
            elif ud_ratio >= 0.58 and lr_ratio > 0.42 and lr_ratio <= 0.57:
                iris_position = "down"
            elif ud_ratio <= 0.42 and lr_ratio > 0.42 and lr_ratio <= 0.57:
                iris_position = "up"    
            elif lr_ratio <= 0.42:
                iris_position = "right"
            elif lr_ratio > 0.57:
                iris_position = "left"

            return iris_position, lr_ratio, ud_ratio

        #cap = cv.VideoCapture(0)

        flag = 0
        i = 0
        first_blink = False
        check_second_blink = False


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
                    # print(results.multi_face_landmarks[0].landmark)
                    mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
                    # print(mesh_points.shape)
                    # cv.polylines(frame, [mesh_points[LEFT_EYE]], True, (0, 255, 0), 1, cv.LINE_AA)
                    # cv.polylines(frame, [mesh_points[RIGHT_EYE]], True, (0, 255, 0), 1, cv.LINE_AA)
                    # cv.polylines(frame, [mesh_points[LEFT_IRIS]], True, (0, 0, 255), 1, cv.LINE_AA)
                    # cv.polylines(frame, [mesh_points[RIGHT_IRIS]], True, (0, 0, 255), 1, cv.LINE_AA)
                    (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
                    (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                    center_left = np.array([l_cx, l_cy], dtype = np.int32)
                    center_right = np.array([r_cx, r_cy], dtype = np.int32)
                    cv.circle(frame, center_left, 1, (255,0,255), 1, cv.LINE_AA)
                    cv.circle(frame, center_right, 1, (255,0,255), 1, cv.LINE_AA)
                    cv.circle(frame, mesh_points[R_H_RIGHT][0], 2, (0,255,0), 1, cv.LINE_AA)
                    cv.circle(frame, mesh_points[R_H_LEFT][0], 2, (0,255,0), 1, cv.LINE_AA)
                    cv.circle(frame, mesh_points[R_H_UP][0], 2, (0,255,0), 1, cv.LINE_AA)
                    cv.circle(frame, mesh_points[R_H_DOWN][0], 2, (0,255,0), 1, cv.LINE_AA)
                    cv.circle(frame, mesh_points[L_H_RIGHT][0], 2, (0,255,0), 1, cv.LINE_AA)
                    cv.circle(frame, mesh_points[L_H_LEFT][0], 2, (0,255,0), 1, cv.LINE_AA)
                    cv.circle(frame, mesh_points[L_H_UP][0], 2, (0,255,0), 1, cv.LINE_AA)
                    cv.circle(frame, mesh_points[L_H_DOWN][0], 2, (0,255,0), 1, cv.LINE_AA)

                    iris_pos, lr_ratio, ud_ratio = iris_position(center_right, mesh_points[R_H_RIGHT][0], mesh_points[R_H_LEFT][0], mesh_points[R_H_UP][0], mesh_points[R_H_DOWN][0])
                    total_distance = euclidean_distance(mesh_points[R_H_UP][0], mesh_points[R_H_DOWN][0])
                    #print(total_distance)
                    if total_distance <= 3:
                        
                        if flag == 1:
                            cemal = 0
                        else:
                            i = i + 1
                            print("blink count:",i," command:",iris_pos)
                            flag = 1

                            if first_blink == False:
                                first_blink = True
                                first_blink_time = time.time()
                            elif check_second_blink:
                                first_blink = False
                                check_second_blink = False
                                print("Two blink detected")
                    else:
                        flag = 0
                        if first_blink:
                            check_second_blink = True

                    if first_blink and time.time() - first_blink_time > 5:
                        first_blink = False
                        check_second_blink = False
                        print("Too old.")

                    cv.putText(frame, f"Iris pos: {iris_pos} {lr_ratio:.2f} {ud_ratio:.2f}", (30,30), cv.FONT_HERSHEY_PLAIN, 1.2, (0,255,0), 1, cv.LINE_AA)
                return ret, frame
                #cv.imshow('Frame',frame)
                key = cv.waitKey(1)
                if key == ord('q'):
                    break

        cap.release()
        cv.destroyAllWindows()

