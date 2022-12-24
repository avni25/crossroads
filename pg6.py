import cv2 as cv
import numpy as np
import math



cap = cv.VideoCapture("videos/intersection-8-720-b.mp4") 
obj_detector = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=50)



isSelected = False
roi = []
COUNT = 0
OFFSET = 3                 # cizgilerin etrafinda olusturalacak olan dortgenin genisliğini belirlemede kullanılacak
DISTANCE_PER_FRAME = 30     # Object Tracking icin noktanin onceki ile simdiki konumları arasındaki max fark
VIDEO_SPEED_RATE = 25       # video oynatma hızı
CONTOUR_AREA_LIMIT = 1000   # tespit edilen contour lari filtrelemek icin alan limiti uygulanmıstır
detections = {}             # tespit edilen arabalar id ile birlikte dictionary olarak tutar
track_id = 0                # track id her tespit edilen ve hareket eden araba icin atanır.
contours_center_current = []    # suan ki frame de tespit edilen contour ların orta noktalarını tutat dizi
contours_center_prev = [] 
COLORS = [      (152, 60, 125),     # purple
                (53, 67, 203),      
                (71, 55, 40), 
                (0, 84, 211),
                (199, 153, 84),
                (46, 58, 176),
                (51, 255, 230)      # 6-yellow
            ]

CARS = {}

def get_center(x,y,w,h):
    return x + int(w/2), y + int(h/2)


while True:
    ret, frame = cap.read()
    frame2 = frame.copy()
    if not isSelected:
        roi = list(cv.selectROI("qwe", frame))        
        print(roi)
        print(len(roi))
        isSelected = True
    
    x1 = roi[0]
    y1 = roi[1]
    x2= roi[0] + roi[2]
    y2 = roi[1] + roi[3]  

    if roi[2] > roi[3]:
        y2 = y1
    else:
        x2 = x1 

    COUNT += 1
    kernel = np.ones((8, 8), np.uint8)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)    # frame i grayscale a cevirir
    blurred = cv.GaussianBlur(gray, (5,5), 0)       # grayscalei bulanıklastırır
    mask = obj_detector.apply(blurred)              # bulanık framei maskeler. siyah beyaz goruntu cıkar
    dilated = cv.dilate(mask, kernel, iterations=1) # maskelenmis framede beyazla tespit edilen objeleri genisletir
    _, th = cv.threshold(mask, 254, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)    # koseleri olan objeleri tespit eder

    

    for(i, c) in enumerate(contours):   # tespit edilen herbir contour(obje) iterasyona alınarak islem yapılır
        (x, y, w, h) = cv.boundingRect(c) # contour un x,y,w,h koordinatlarını dondurur
        contour_valid = (w >= 30) and (     # eger genisliği 30 px den kucukse isleme almaz/atlar
            h >= 30)
        
        center = get_center(x,y,w,h) # contourun orta noktasını dondurur

        if not contour_valid:
            continue  
        area = cv.contourArea(c) # contourun alanını hesaplar
        
        if isSelected:            
            try:
                cv.line(frame, (x1, y1), (x2, y2), COLORS[2], 20)        
            except:
                pass
        # contour alanı belirlenen limitten buyukse listeye ekler. araba olma ihtimali vardir
        if area > CONTOUR_AREA_LIMIT:   
            contours_center_current.append(center)  
        else:
            continue
    # -------------------------------------------------------------------------------
    if COUNT <= 1:        
        for pt in contours_center_current:
            for pt2 in contours_center_prev:
                distance = math.hypot(pt2[0]-pt[0], pt2[1]-pt[1])
                if distance < DISTANCE_PER_FRAME:
                    detections[track_id] = pt
                    track_id += 1
                # if distance == 0:
                #     try:
                #         detections.pop(track_id)
                #     except:
                #         print("")
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
    # ----------------------------------------------------------------------------
    for object_id, pt in detections.items():
        cv.rectangle(frame, (pt[0]-20, pt[1]-20), (pt[0]+20,pt[1]+20), (0, 255, 0), 2)
        cv.circle(frame, pt, 5, (0, 255, 0), -1)
        cv.rectangle(frame, (pt[0]-20, pt[1]-32),(pt[0]+10, pt[1]-20), COLORS[0], -1)
        cv.putText(frame, str(object_id), (pt[0]-20, pt[1]-20),cv.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[6], 2) 
        
        if roi[2] > roi[3]: # if line is horizontal
            if (pt[1] > roi[1]-OFFSET and pt[1] < roi[1]+roi[3]+OFFSET) and (pt[0] > roi[0] and pt[0] < roi[0]+roi[2]):
                CARS[str(object_id)] = pt 
        else:   # if line is vertical
            if (pt[0] < (roi[0]+roi[2]+ OFFSET) and pt[0] > roi[0] - OFFSET) and (pt[1]>roi[1] and pt[1] < roi[1]+roi[3]):
                CARS[str(object_id)] = pt
    
    contours_center_prev = contours_center_current.copy()

    cv.putText(frame, str(len(CARS)), (x2,y2), cv.FONT_HERSHEY_SIMPLEX, 1.0, COLORS[6], 3) 

    alpha = 0.3
    frame2 = cv.addWeighted(frame2, alpha, frame, 1-alpha, 0)

    # cv.imshow("fr",frame)
    cv.imshow("fr2",frame2)

    if cv.waitKey(25) & 0xFF == ord('q'):
        break


cap.release() 
cv.destroyAllWindows()








