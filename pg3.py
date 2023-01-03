import cv2 as cv
import numpy as np
import math


OFFSET = 3                  # cizgilerin etrafinda olusturalacak olan dortgenin genisliğini belirlemede kullanılacak
DISTANCE_PER_FRAME = 30     # Object Tracking icin noktanin onceki ile simdiki konumları arasındaki max fark
VIDEO_SPEED_RATE = 25       # video oynatma hızı
CONTOUR_AREA_LIMIT = 1200   # tespit edilen contour lari filtrelemek icin alan limiti uygulanmıstır
detections = {}             # tespit edilen arabalar id ile birlikte dictionary olarak tutar
track_id = 0                # track id her tespit edilen ve hareket eden araba icin atanır.
contours_center_current = []    # suan ki frame de tespit edilen contour ların orta noktalarını tutat dizi
contours_center_prev = []       # bir onceki frame de tespit edilen contour ların orta noktalarını tutat dizi
COUNT = 0                   # ilk frame i gecmek icin kullanılmıstır. cunku ilk framde bir onceki noktalarını tutan dizi bostur


CARS_LINE_1 = {}        # Line 1 den gecen arabalrı id ve orta noktalarını saklar
CARS_LINE_1_R = {}      # Line 1R den gecen arabalrı id ve orta noktalarını saklar
CARS_LINE_2={}          # Line 2 den gecen arabalrı id ve orta noktalarını saklar
CARS_LINE_2_L={}        # Line 2L den gecen arabalrı id ve orta noktalarını saklar
CARS_LINE_3={}          # Line 3 den gecen arabalrı id ve orta noktalarını saklar
CARS_LINE_3_R={}        # Line 3R den gecen arabalrı id ve orta noktalarını saklar
CARS_LINE_4={}          # Line 4 den gecen arabalrı id ve orta noktalarını saklar
CARS_LINE_5={}          # Line 5 den gecen arabalrı id ve orta noktalarını saklar


# lines coordinates
L1 = [510, 600, 600, 600]   # L1 arabaların sayılacagi cizginin koordinatlari
L1_R = [610, 600, 800, 600] # L1R arabaların sayılacagi cizginin koordinatlari
L2 = [350, 230, 350, 310]   # L2 arabaların sayılacagi cizginin koordinatlari
L2_L = [350, 320, 350, 480] # L2L arabaların sayılacagi cizginin koordinatlari
L3 = [510, 100, 670, 100]   # L3 arabaların sayılacagi cizginin koordinatlari
L3_R = [680, 100, 750, 100] # L3r arabaların sayılacagi cizginin koordinatlari
L4 = [880, 200, 880, 350]   # L4 arabaların sayılacagi cizginin koordinatlari
L5 = [880, 370, 880, 500]   # L5 arabaların sayılacagi cizginin koordinatlari

lines = [L1, L1_R, L2, L2_L, L3, L3_R, L4, L5]

COLORS = [      (152, 60, 125),          # purple
                (53, 67, 203),      
                (71, 55, 40), 
                (0, 84, 211),
                (199, 153, 84),
                (46, 58, 176),
                (51, 255, 230)      # 6-yellow
            ]




cap = cv.VideoCapture("videos/intersection-8-720-b.mp4")  # videoyou dondurur
obj_detector = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=50)
# tracker = cv.legacy.MultiTracker_create()

'''
    dortgenin orta noktasini bulur, x, y degerlerinin dondurur
'''
def get_center(x,y,w,h):
    return x + int(w/2), y + int(h/2)

'''
    iki noktasi verilen dogrunun eğimini hesaplar
'''
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

def draw_panel(frame):
    cv.rectangle(frame, (190, 10), (350, 200), COLORS[0], -1)
    cv.putText(frame, "Accuracy", (250, 20),cv.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[6], 1)
    cv.putText(frame, "L1:      0.98", (200, 40),cv.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[6], 1)
    cv.putText(frame, "L1_R:    0.97", (200, 60),cv.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[6], 1)
    cv.putText(frame, "L2:      0.92", (200, 80),cv.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[6], 1)
    cv.putText(frame, "L2_L:    0.67", (200, 100),cv.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[6], 1)
    cv.putText(frame, "L3:      0.94", (200, 120),cv.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[6], 1)
    cv.putText(frame, "L3_L:    0.79", (200, 140),cv.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[6], 1)
    cv.putText(frame, "L4:      0.6", (200, 160),cv.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[6], 1)
    cv.putText(frame, "L5:      0.96", (200, 180),cv.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[6], 1)


def draw_lines2(frame):
    color = COLORS[4]
    for i, line  in enumerate(lines):
        if i%2 == 0:
            color = COLORS[2]
        else:
            color = COLORS[4]
        cv.line(frame, (line[0], line[1]), (line[2], line[3]), color, 20)

while True:
    timer = cv.getTickCount()
    ret, frame = cap.read()
    # (h,w, _) = frame.shape
    # print(f'{h} - {w}')
    COUNT += 1
    # extract region of Interest
    # roi = frame[200: 600, 300: 900]
    # roi2 = frame[200:650, 200:1200]
    # ------------------------IMAGE PRE-PROCESSING-------------------------------------
    kernel = np.ones((8, 8), np.uint8)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)    # frame i grayscale a cevirir
    blurred = cv.GaussianBlur(gray, (5,5), 0)       # grayscalei bulanıklastırır
    mask = obj_detector.apply(blurred)              # bulanık framei maskeler. siyah beyaz goruntu cıkar
    # ced = cv.Canny(mask, 50, 150)                # bulanık framei edge detection yapar
    # filtered = cv.fastNlMeansDenoising(mask, None, 20, 7, 21) 
    # erosion = cv.erode(mask, kernel, iterations=1)
    dilated = cv.dilate(mask, kernel, iterations=1) # maskelenmis framede beyazla tespit edilen objeleri genisletir
    # _, dilated = cv.threshold(dilated, 254, 255, cv.THRESH_BINARY)
    _, th = cv.threshold(mask, 254, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)    # koseleri olan objeleri tespit eder
    # cv.drawContours(frame, contours, -1, (0,255,0), 3)
    
    # info panel dogruluk oranlarını gosterir.
    # draw_panel(frame)
    draw_lines2(frame)
    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------ 
    for contour in contours:   # tespit edilen herbir contour(obje) iterasyona alınarak islem yapılır
        (x, y, w, h) = cv.boundingRect(contour) # contour un x,y,w,h koordinatlarını dondurur
        contour_valid = (w >= 30) and (h >= 30)   # eger genisliği 30 px den kucukse isleme almaz/atlar
        
        if not contour_valid:
            continue 
         
        center = get_center(x,y,w,h) # contourun orta noktasını dondurur
        area = cv.contourArea(contour) # contourun alanını hesaplar        
        # cv.rectangle(frame, (x, y), (x+w,y+h), (0, 255, 0), 2)

        # contour alanı belirlenen limitten buyukse listeye ekler. araba olma ihtimali vardir
        if area > CONTOUR_AREA_LIMIT:   
            contours_center_current.append(center)  
            # cv.rectangle(frame, (x,y), (x+w, y+h), COLORS[0], 2)
        else:
            continue
    # -----------------------------OBJECT TRACKING--------------------------------------------------
    
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
    
    contours_center_prev = contours_center_current.copy()
    
    # ----------------------------------------------------------------------------
    for object_id, pt in detections.items():
        cv.rectangle(frame, (pt[0]-20, pt[1]-20), (pt[0]+20,pt[1]+20), (0, 255, 0), 2)
        cv.circle(frame, pt, 5, (0, 255, 0), -1)
        cv.rectangle(frame, (pt[0]-20, pt[1]-32),(pt[0]+10, pt[1]-20), COLORS[0], -1)
        cv.putText(frame, str(object_id), (pt[0]-20, pt[1]-20),cv.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[6], 2) 
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
    

    cv.putText(frame, str(len(CARS_LINE_1)), ((L1[0]+((L1[2]-L1[0])//2)), L1[1]+5), cv.FONT_HERSHEY_SIMPLEX, 0.7, (188, 222, 17), 2)    
    cv.putText(frame, "L1", (L1[0], L1[1]+5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (51, 255, 255), 2)  
    cv.putText(frame, str(len(CARS_LINE_1_R)), ((L1_R[0]+((L1_R[2]-L1_R[0])//2))+10, L1_R[1]+5), cv.FONT_HERSHEY_SIMPLEX, 0.7, (188, 222, 17), 2)    
    cv.putText(frame, "L1_R", (L1_R[0], L1_R[1]+5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (51, 255, 255), 2)  
    cv.putText(frame, str(len(CARS_LINE_2)), (L2[0]-5, (L2[1]+((L2[3]-L2[1])//2))), cv.FONT_HERSHEY_SIMPLEX, 0.7, (188, 222, 17), 2)
    cv.putText(frame, "L2", (L2[0]-10, L2[1]+5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (51, 255, 255), 2)  
    cv.putText(frame, str(len(CARS_LINE_2_L)), (L2_L[0]-5, (L2_L[1]+((L2_L[3]-L2_L[1])//2))), cv.FONT_HERSHEY_SIMPLEX, 0.7, (188, 222, 17), 2)
    cv.putText(frame, "L2_L", (L2_L[0]-10, L2_L[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (51, 255, 255), 2)  
    cv.putText(frame, str(len(CARS_LINE_3)), (640, L3[1]+5), cv.FONT_HERSHEY_SIMPLEX, 0.7, (188, 222, 17), 2)
    cv.putText(frame, "L3", (L3[0], L3[1]+5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (51, 255, 255), 2)  
    cv.putText(frame, str(len(CARS_LINE_3_R)), ((L3_R[0]+((L3_R[2]-L3_R[0])//2))+10, L3_R[1]+5), cv.FONT_HERSHEY_SIMPLEX, 0.7, (188, 222, 17), 2)
    cv.putText(frame, "L3_R", (L3_R[0], L3_R[1]+5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (51, 255, 255), 2)  
    cv.putText(frame, str(len(CARS_LINE_4)), (L4[0]-5, 270), cv.FONT_HERSHEY_SIMPLEX, 0.7, (188, 222, 17), 2)
    cv.putText(frame, "L4", (L4[0]-10, L4[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (51, 255, 255), 2)  
    cv.putText(frame, str(len(CARS_LINE_5)), (L5[0]-5, 430), cv.FONT_HERSHEY_SIMPLEX, 0.7, (188, 222, 17), 2)
    cv.putText(frame, "L5", (L5[0]-5, L5[1]+5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (51, 255, 255), 2)  

    fps = int(cv.getTickFrequency()/(cv.getTickCount() - timer))    
    cv.putText(frame, f'fps: {str(fps)}', (350,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (51, 255, 255), 2)

    cv.imshow("Frame", frame)
    # cv.imshow("roi", roi2)
    # cv.imshow("Mask", mask)
    # cv.imshow("filtered", filtered)
    # cv.imshow("dilated",dilated)
    # cv.imshow("erosion",erosion)


    if cv.waitKey(VIDEO_SPEED_RATE) & 0xFF == ord('q'):
        break


cap.release() 
cv.destroyAllWindows()






