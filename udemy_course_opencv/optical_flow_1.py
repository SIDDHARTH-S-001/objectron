import cv2
from cv2 import calcOpticalFlowFarneback
from cv2 import calcOpticalFlowPyrLK
from cv2 import add
import numpy as np

corner_track_params = dict(maxCorners = 10, qualityLevel = 0.3, minDistance = 7, blockSize = 7)
lk_params = dict(winSize = (200,200), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)) # for lucas canade method, uses sometging called image processing pyramid, the maxLevel decides the level.
# number of iterations = 10 (more number runs a more exhaustive search)
# 0.03 is the episilon value (small value finish search faster) basically we have to trade off between speed and accuracy

cap = cv2.VideoCapture(0)
success,prev_frame = cap.read()
prev_grey = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prevPts = cv2.goodFeaturesToTrack(prev_grey, mask=None,**corner_track_params)
mask = np.zeros_like(prev_frame)

while True:
    success, frame = cap.read()
    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    nextPts, status, err = cv2.calcOpticalFlowPyrLK(prev_grey, frame_grey, prevPts, None, **lk_params )

    good_new = nextPts[status==1]
    good_prev = prevPts[status==1]

    for i, (new, prev) in enumerate(zip(good_new, good_prev)):
        x_new, y_new = new.ravel()
        x_prev, y_prev = prev.ravel()
        
        x_new = int(x_new)
        y_new = int(y_new)
        x_prev = int(x_prev)
        y_prev = int(y_prev)

        mask = cv2.line(mask, (x_new, y_new), (x_prev, y_prev), (0, 255, 0), 3) #requires pts1 and pts2 as integers
        frame = cv2.circle(mask, (x_new, y_new), 4,  (0, 0, 255), -1)
    
    img = add(frame, mask)
    cv2.imshow('tracking', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_grey = frame_grey.copy()
    prevPts = good_new.reshape(-1, 1, 2)

