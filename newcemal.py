import tensorflow as tf 
import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import time

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

cap = cv.VideoCapture(0)

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
            print(total_distance)
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
                    print("Two blink detected")
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
            cv.putText(frame, f"Iris pos: {iris_pos} {lr_ratio:.2f} {ud_ratio:.2f}", (30,30), cv.FONT_HERSHEY_PLAIN, 1.2, (0,255,0), 1, cv.LINE_AA)

        cv.imshow('Frame',frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break

cap.release()
cv.destroyAllWindows()


