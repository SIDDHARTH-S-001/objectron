import cv2
import numpy as np

corner_track_params = dict(maxCorners = 10, qualityLevel = 0.3, minDistance = 7, blockSize = 7)
lk_params = dict(winSize = (200,200), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

cap = cv2.VideoCapture(0)
success,prev_frame = cap.read()

prev_grey = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

hsv_mask = np.zeros_like(prev_frame)
# print(hsv_mask.shape) == (480, 640, 3)
hsv_mask[:,:,1] = 255

while True:
    success, frame = cap.read()
    next_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_grey, next_img, None, pyr_scale=0.5, levels= 3, winsize=15, iterations=5,poly_n=5, poly_sigma= 1.2, flags=0 )
    
    # method used -> tracking object using a vector by comparing position in current frame and previous frame
    # converting to HSV wherein angle represents hue value (setting to 180 degrees instead of 360)
    # the magnitude of vector represents value which is normalized between 0 and 255

    mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1], angleInDegrees=True)

    hsv_mask[:,:,0] = ang/2 # hue

    hsv_mask[:,:,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    bgr = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

    cv2.imshow('frame', bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_grey = next_img.copy()
