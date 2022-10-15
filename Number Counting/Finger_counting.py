import cv2
import mediapipe as mp
import time
import os

cap= cv2.VideoCapture(0)

mpHands= mp.solutions.hands
hands = mpHands.Hands(False)
mpDraw= mp.solutions.drawing_utils

folder_path= r'C:\Users\shailesh suthar\Downloads\hand_pic'
my_list = os.listdir(folder_path)
print(my_list)
overlay_list=[]

for img_path in my_list:
    image= cv2.imread(f'{folder_path}/{img_path}')
    image= cv2.resize(image, (100,150))
    overlay_list.append(image)

# print(len(overlay_list))

tip_ids= [4,8,12,16,20]

while True:
    fingure= []
    lm_list= []
    success, img= cap.read()
    imgRGB= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results= hands.process(imgRGB)
    

    if results.multi_hand_landmarks:
        x_list, y_list = [], []

        for i in results.multi_handedness:
                for index, score in enumerate(i.classification):
                    hand_id= score.index
        for handlms in results.multi_hand_landmarks:
            for id, lm in enumerate(handlms.landmark):
                # print(id, lm)
                h, w, c= img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx, cy)
                lm_list.append([id, cx, cy])
                x_list.append(cx)
                y_list.append(cy)
                x_min, x_max = min(x_list), max(x_list)
                y_min, y_max = min(y_list), max(y_list)
                
            # print(results.multi_handedness)
            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)
    # print(lm_list)
        if len(lm_list)!=0:
            if hand_id==1:
                if lm_list[tip_ids[0]][1] < lm_list[tip_ids[0]-1][1]:
                    fingure.append(1)
                else:
                    fingure.append(0)
            if hand_id==0:
                if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0]-1][1]:
                    fingure.append(1)
                else:
                    fingure.append(0)
            for id in range(1,len(tip_ids)):
                if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id]-2][2]:
                    fingure.append(1)
                else:
                    fingure.append(0)
        # print(fingure)
        # print(lm_list)
        total_fingers= fingure.count(1)
        h, w, c = overlay_list[total_fingers].shape
        img[0:h , 0:w] = overlay_list[total_fingers]
        cv2.rectangle(img, (x_min-20, y_min-20), (x_max+20, y_max+20), (255,0,255),3)
        cv2.rectangle(img, (x_min-22, y_min-70), (x_max+22, y_min-20), (255,0,255),cv2.FILLED)
        cv2.putText(img, f'Count: {total_fingers}', (x_min-10,y_min-30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255),2)

    cv2.imshow('image',img)
    k= cv2.waitKey(1)
    if k == 27:
        break
cv2.destroyAllWindows()
