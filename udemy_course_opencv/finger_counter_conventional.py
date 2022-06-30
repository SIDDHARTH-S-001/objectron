from audioop import avg
from turtle import back, distance
import cv2
from cv2 import THRESH_BINARY
import numpy as np
from sklearn.metrics import pairwise

background = None

accumulated_weight = 0.5

roi_top = 20
roi_bottom  = 300
roi_right = 300
roi_left = 600

def calc_accum_avg(frame, accumulated_weight):
    global background
    if background is None:
        background = frame.copy().astype('float')
        return None
    cv2.accumulateWeighted(frame, background, accumulated_weight)

def segment(frame, threshold_min = 25):
    diff = cv2.absdiff(background.astype('uint8'),frame)
    ret, thresholded = cv2.threshold(diff, threshold_min, 255, THRESH_BINARY)
    image, contours = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)==0:
        return None
    else:
        # assuming largest external contour is the hand
        hand_segment = max(contours, key=cv2.contourArea)

        return (thresholded, hand_segment)

def count_fingers(thresholded, hand_segment):
    conv_hull = cv2.convexHull(hand_segment)

    top = tuple(conv_hull[conv_hull][:, :, 1].argmin()[0])
    bottom = tuple(conv_hull[conv_hull][:, :, 1].argmin()[0])
    right = tuple(conv_hull[conv_hull][:, :, 0].argmin()[0])
    left = tuple(conv_hull[conv_hull][:, :, 0].argmin()[0])

    cx = (left[0] + right[0])//2
    cy = (top[1] + bottom[1])//2

    distance = pairwise.euclidean_distances([cx, cy], Y=[left, right, top, bottom])[0]
    max_distance = distance.max()
    radius = int(0.9*max_distance)
    circumference = (2*np.pi*radius)
    circular_roi = np.zeros(thresholded[:2],dtype='unit8')
    cv2.circle(circular_roi, (cx, cy), radius, 255, 10)
    image, contours = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0

    for cnt in contours:
        (x,y,w,h) = cv2.boundingRect(cnt)
        out_of_wrist = (cy + (cy*0.25)) > (y+h)
        limit_pts = ((circumference*0.25)>cnt.shape[0])
        if out_of_wrist and limit_pts:
            count+=1
    return count 

cap = cv2.VideoCapture(0)
num_frames = 0
while True:
    success, frame = cap.read()
    frame_copy = frame.copy()
    roi = frame[roi_top:roi_bottom, roi_right:roi_left]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.GaussianBlur(roi, (7,7), 0)

    if num_frames < 60:
        calc_accum_avg(roi, accumulated_weight)

    if num_frames<=59:
        cv2.putText(roi, 'Getting background', (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.imshow('finger_count', frame_copy)
    else:
        hand = segment(roi)
        if hand is not None:
            thresholded, hand_segment = hand
            cv2.drawContours(frame_copy, [hand_segment+(roi_right, roi_top)], -1, (0,255, 0), 3)

            fingers_count = count_fingers(thresholded, hand_segment)

            cv2.putText(frame_copy, str(fingers_count), (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            cv2.imshow('thresholded', thresholded)

    cv2.rectangle(frame_copy, (roi_left, roi_right), (roi_top, roi_bottom), (255,255,255), 3)
    num_frames += 1
    cv2.imshow('finger_count', frame_copy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


