import cv2
import numpy as np

cap = cv2.VideoCapture(0)
success, frame = cap.read()

face_cascade =  cv2.CascadeClassifier('D:/OPEN CV/udemy_course_opencv/haarcascade_frontalface_alt.xml')
face_rects = face_cascade.detectMultiScale(frame)

(face_x, face_y, w, h) =  tuple(face_rects[0])
track_window = (face_x, face_y, w, h)

roi = frame[face_y:face_y+h, face_x, face_x+w]

hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist(hsv_roi, [0], None, [180], [0, 180])

cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    success, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    success, track_window = cv2.meanShift(dst, track_window, term_crit)

    x,y,w,h = track_window
    img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 5)

    cv2.imshow('frame', img2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break