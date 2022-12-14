import cv2 as cv
import numpy as np
import math


OFFSET = 3  
DISTANCE_PER_FRAME = 30
VIDEO_SPEED_RATE = 25
CONTOUR_AREA_LIMIT = 1000
detections = {}
track_id = 0
contours_center_current = []
contours_center_prev = []
count = 0


CARS_LINE_1 = {}
CARS_LINE_1_R = {}
CARS_LINE_2={}
CARS_LINE_2_L={}
CARS_LINE_3={}
CARS_LINE_3_R={}
CARS_LINE_4={}
CARS_LINE_5={}
CARS_LINE_6={}


# lines coordinates
L1 = [510, 600, 600, 600]
L1_R = [610, 600, 800, 600] 
L2 = [350, 230, 350, 310]
L2_L = [350, 320, 350, 480]
L3 = [510, 100, 670, 100]
L3_R = [680, 100, 750, 100]
L4 = [880, 200, 880, 350]
L5 = [880, 370, 880, 500]


TEXT_COLOR = ()




cap = cv.VideoCapture("videos/intersection-8-720.mp4")
obj_detector = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=50)
tracker = cv.legacy.MultiTracker_create()

def get_center(x,y,w,h):
    return x + int(w/2), y + int(h/2)

def calculate_line_slope(x1,y1,x2,y2):
    return (y2-y1)/(x2-x1)

'''
if m negative and result negative point is on the right side of the line
if m positive and result positive point is on the right side of the line
'''
def calculate_distance_between_point_and_line(x, y, x1, y1, x2, y2):
    m = calculate_line_slope(x1,y1,x2,y2)
    f = (m*x) - (m*x2) - y + y2
    s = math.sqrt(m**2 + 1)    
    return f / s




while True:
    ret, frame = cap.read()
    (h,w, _) = frame.shape
    # print(f'{h} - {w}')
    count += 1
    # extract region of Interest
    roi = frame[200: 600, 300: 900]
    roi2 = frame[200:650, 200:1200]
    kernel = np.ones((8, 8), np.uint8)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5,5), 0)
    ced = cv.Canny(blurred, 50, 150)
    mask = obj_detector.apply(blurred)
    # filtered = cv.fastNlMeansDenoising(mask, None, 20, 7, 21) 
    dilated = cv.dilate(mask, kernel, iterations=1)
    # _, dilated = cv.threshold(dilated, 254, 255, cv.THRESH_BINARY)
    _, th = cv.threshold(mask, 254, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(frame, contours, -1, (0,255,0), 3)
     
    for(i, c) in enumerate(contours):
        (x, y, w, h) = cv.boundingRect(c)
        contour_valid = (w >= 30) and (
            h >= 30)

        center = get_center(x,y,w,h)

        if not contour_valid:
            continue  
        area = cv.contourArea(c)
        # print(area)
        
        cv.line(frame, (L1[0], L1[1]), (L1[2], L1[3]), (0, 0, 255), 10)
        cv.line(frame, (L1_R[0], L1_R[1]), (L1_R[2], L1_R[3]), (255, 0, 0), 10)
        cv.line(frame, (L2[0], L2[1]), (L2[2], L2[3]), (0, 0, 255), 10)
        cv.line(frame, (L2_L[0], L2_L[1]), (L2_L[2], L2_L[3]), (255, 0, 0), 10)
        cv.line(frame, (L3[0], L3[1]), (L3[2], L3[3]), (0, 0, 255), 10)
        cv.line(frame, (L3_R[0], L3_R[1]), (L3_R[2], L3_R[3]), (255, 0, 0), 10)
        cv.line(frame, (L4[0], L4[1]), (L4[2], L4[3]), (0, 0, 255), 10)
        cv.line(frame, (L5[0], L5[1]), (L5[2], L5[3]), (255, 0, 0), 10)

        if area > CONTOUR_AREA_LIMIT:
            contours_center_current.append(center)           
            # contours_center_current.remove(center)
    if count <= 2:        
        for pt in contours_center_current:
            for pt2 in contours_center_prev:
                distance = math.hypot(pt2[0]-pt[0], pt2[1]-pt[1])
                if distance < DISTANCE_PER_FRAME:
                    detections[track_id] = pt
                    track_id += 1
    else:
        detections_copy =  detections.copy()
        contours_center_current_copy = contours_center_current.copy()
        for object_id, pt2 in detections_copy.items():
            obj_exists = False
            for pt in contours_center_current_copy:
                distance = math.hypot(pt2[0]-pt[0], pt2[1]-pt[1])
                if distance < DISTANCE_PER_FRAME:
                    detections[object_id] = pt
                    obj_exists = True
                    if pt in contours_center_current:
                        contours_center_current.remove(pt)
                    continue
            if not obj_exists:
                detections.pop(object_id)

        for pt in contours_center_current:
            detections[track_id] = pt 
            track_id +=1

    for object_id, pt in detections.items():
        cv.rectangle(frame, (pt[0]-20, pt[1]-20), (pt[0]+20,pt[1]+20), (0, 255, 0), 2)
        cv.circle(frame, pt, 5, (0, 255, 0), -1)
        cv.putText(frame, str(object_id), (pt[0], pt[1]-10),cv.FONT_HERSHEY_SIMPLEX, 0.7, (188, 222, 17), 2) 
        if (pt[1] > L1[1]-OFFSET and pt[1] < L1[3]+OFFSET) and (pt[0] > L1[0] and pt[0] < L1[2]):
            CARS_LINE_1[str(object_id)] = pt 
        if (pt[1] > L1_R[1]-OFFSET and pt[1] < L1_R[3]+OFFSET) and (pt[0] > L1_R[0] and pt[0] < L1_R[2]):
            CARS_LINE_1_R[str(object_id)] = pt  
        if (pt[0] < L2[0] + OFFSET and pt[0] > L2[2] - OFFSET) and (pt[1]>L2[1] and pt[1]<L2[3]):
            CARS_LINE_2[str(object_id)] = pt
        if (pt[0] < L2_L[0] + OFFSET+10 and pt[0] > L2_L[2] - OFFSET) and (pt[1]>L2_L[1] and pt[1]<L2_L[3]):
            CARS_LINE_2_L[str(object_id)] = pt
        if (pt[1] > L3[1]-OFFSET and pt[1] < L3[3]+OFFSET) and (pt[0] > L3[0] and pt[0] < L3[2]):
            CARS_LINE_3[str(object_id)] = pt
        if (pt[1] > L3_R[1]-OFFSET and pt[1] < L3_R[3]+OFFSET+20) and (pt[0] > L3_R[0] and pt[0] < L3_R[2]):
            CARS_LINE_3_R[str(object_id)] = pt    
        if (pt[0] < L4[0]+ OFFSET and pt[0] > L4[2] - OFFSET) and (pt[1]>L4[1] and pt[1]<L4[3]):
            CARS_LINE_4[str(object_id)] = pt
        if (pt[0] < L5[0] + OFFSET+10 and pt[0] > L5[2] - OFFSET) and (pt[1]>L5[1] and pt[1]<L5[3]):
            CARS_LINE_5[str(object_id)] = pt
    
    contours_center_prev = contours_center_current.copy()

    cv.putText(frame, str(len(CARS_LINE_1)), ((L1[0]+((L1[2]-L1[0])//2)), L1[1]), cv.FONT_HERSHEY_SIMPLEX, 0.7, (188, 222, 17), 2)    
    cv.putText(frame, "L1", (L1[0], L1[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (51, 255, 255), 2)  
    cv.putText(frame, str(len(CARS_LINE_1_R)), ((L1_R[0]+((L1_R[2]-L1_R[0])//2)), L1_R[1]), cv.FONT_HERSHEY_SIMPLEX, 0.7, (188, 222, 17), 2)    
    cv.putText(frame, "L1_R", (L1_R[0], L1_R[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (51, 255, 255), 2)  
    cv.putText(frame, str(len(CARS_LINE_2)), (L2[0], (L2[1]+((L2[3]-L2[1])//2))), cv.FONT_HERSHEY_SIMPLEX, 0.7, (188, 222, 17), 2)
    cv.putText(frame, "L2", (L2[0], L2[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (51, 255, 255), 2)  
    cv.putText(frame, str(len(CARS_LINE_2_L)), (L2_L[0], (L2_L[1]+((L2_L[3]-L2_L[1])//2))), cv.FONT_HERSHEY_SIMPLEX, 0.7, (188, 222, 17), 2)
    cv.putText(frame, "L2_L", (L2_L[0], L2_L[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (51, 255, 255), 2)  
    cv.putText(frame, str(len(CARS_LINE_3)), (640, L3[1]), cv.FONT_HERSHEY_SIMPLEX, 0.7, (188, 222, 17), 2)
    cv.putText(frame, "L3", (L3[0], L3[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (51, 255, 255), 2)  
    cv.putText(frame, str(len(CARS_LINE_3_R)), ((L3_R[0]+((L3_R[2]-L3_R[0])//2)), L3_R[1]), cv.FONT_HERSHEY_SIMPLEX, 0.7, (188, 222, 17), 2)
    cv.putText(frame, "L3_R", (L3_R[0], L3_R[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (51, 255, 255), 2)  
    cv.putText(frame, str(len(CARS_LINE_4)), (L4[0], 270), cv.FONT_HERSHEY_SIMPLEX, 0.7, (188, 222, 17), 2)
    cv.putText(frame, "L4", (L4[0], L4[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (51, 255, 255), 2)  
    cv.putText(frame, str(len(CARS_LINE_5)), (L5[0], 430), cv.FONT_HERSHEY_SIMPLEX, 0.7, (188, 222, 17), 2)
    cv.putText(frame, "L5", (L5[0], L5[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (51, 255, 255), 2)  

    cv.imshow("Frame", frame)
    # cv.imshow("roi", roi2)
    # cv.imshow("Mask", mask)
    # cv.imshow("filtered", filtered)
    # cv.imshow("dilated",dilated)
    


    if cv.waitKey(VIDEO_SPEED_RATE) & 0xFF == ord('q'):
            break


cap.release() 
cv.destroyAllWindows()





