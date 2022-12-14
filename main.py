import cv2 as cv
import numpy as np


VIDEO_SPEED_RATE = 20   # higher value, slower video (25ms is the normal speed value)
VIDEO_URL = "videos/intersection-5.mp4"



OFFSET = 10
LANE_WIDTH = 90
CENTER_LANE = [50,50,1200,650]
V_LANE = [50,700, 930, 150]
CL_RIGHT = [i-LANE_WIDTH for i in CENTER_LANE]
CL_LEFT = [i+LANE_WIDTH for i in CENTER_LANE]

MATCHES = []
CARS_L1=0
CARS_L2=0
CARS_L3=0
CARS_L4=0

L1 =[470, 440, 600, 500] 
L2 = [440, 320, 500, 300]
L3 = [690, 390, 740, 360]

cap = cv.VideoCapture(VIDEO_URL)
car_cascade = cv.CascadeClassifier('cars.xml')

print(cv.__version__)
tracker = cv.legacy_TrackerCSRT.create()

ret, frame = cap.read()
bbox=(400,200, 800, 600)
# bbox = cv.selectROI(frame, False)
ok=tracker.init(frame, bbox)

def center_point(x,y,w,h):
    x1 = int(w / 2)
    y1 = int(h / 2)
 
    cx = x + x1
    cy = y + y1
    return cx, cy


if (cap.isOpened()== False):
    print("Error opening video file")

# Read until video is completed
while(cap.isOpened()):
    ret, frame = cap.read()
    # print(f'ret type: {type(ret)}, frame type: {type(frame)}')
    roi = frame[200: 600, 300: 900]
    
    timer = cv.getTickCount()
    ret, bbox = tracker.update(frame)

    if ret == True:    
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (9,9), 0)
        ced = cv.Canny(blurred, 50, 150)
        ret2, th = cv.threshold(gray, 20, 255, cv.THRESH_BINARY)
        contours, hierarchy = cv.findContours(th, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        
        for cnt in contours:
            cv.drawContours(frame, [cnt], -1, (0,255,0), 2)
        # print(contours)
        # Detects cars of different sizes in the input image
        cars = car_cascade.detectMultiScale(blurred, 1.1, 1)
        # print(cars)

        for(i, c) in enumerate(contours):
            (x, y, w, h) = cv.boundingRect(c)
            contour_valid = (w >= 5) and (
                h >= 5)
        
            if not contour_valid:
                continue    
            
        
            cv.rectangle(roi, (x-10, y-10), (x+w+10, y+h+10), (255, 0, 0), 2)
            cv.line(frame, (L1[0], L1[1]), (L1[2], L1[3]), (0, 0, 255), 2)
            cv.line(frame, (L2[0], L2[1]), (L2[2], L2[3]), (0, 0, 255), 2)
            cv.line(frame, (L3[0], L3[1]), (L3[2], L3[3]), (0, 0, 255), 2)
            cv.circle(frame, (10,10), 5, (255, 0, 0), -1)

            centrolid = center_point(x, y, w, h)
            MATCHES.append(centrolid)
            cv.circle(frame, centrolid, 5, (0, 255, 0), -1)
            cx, cy = center_point(x, y, w, h)
            for (x, y) in MATCHES:
                if (y < (L1[3] + OFFSET) and y > (L1[3] - OFFSET)) and (x < (L1[2] + OFFSET) and x > (L1[2] - OFFSET)):
                    CARS_L1 = CARS_L1+1
                    MATCHES.remove((x, y))
                


        cv.putText(frame, "Total Cars Detected: " + str(CARS_L2), (10, 90), 
                                                        cv.FONT_HERSHEY_SIMPLEX, 1,
                                                        (0, 170, 0), 2)
        cv.imshow("qwe", frame)
        # cv.imshow("qwe", blurred)
        # cv.imshow("th", th)


        if cv.waitKey(VIDEO_SPEED_RATE) & 0xFF == ord('q'):
            break
# When everything done, release
# the video capture object
cap.release()
 
cv.destroyAllWindows()



    
