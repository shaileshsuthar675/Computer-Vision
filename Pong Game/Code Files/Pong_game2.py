import cv2
import numpy as np
import Hand_tracking_module

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4,720)

detector = Hand_tracking_module.handDetector(maxHands=1)

img_background = cv2.imread('Pong Game 2\\background.png')
img_ball = cv2.imread('Pong Game 2\\ball.png', cv2.IMREAD_UNCHANGED)
img_bat1 = cv2.imread('Pong Game 2\\bat1.jpg')
img_game_over = cv2.imread('Pong Game 2\\game_over.jpg')

ball_pos = [50,50]

game_over = False
speedx = 15
speedy = 15
smoothness = 10
score = 0

while True:
    _, img = cap.read()
    img = cv2.flip(img,1)

    
    hands ,img = detector.findHands(img, img_flip=True)
    # print(hands)

    img = cv2.addWeighted(img, 0.2, img_background, 0.8, 0)
    img_ball = cv2.resize(img_ball, (70,70))
    img_bat1 = cv2.resize(img_bat1, (150,30))

    if ball_pos[1]>550:
        game_over = True

    if game_over:
        img = img_game_over
        cv2.putText(img,  str(score).zfill(2), (600,330),  cv2.FONT_HERSHEY_SIMPLEX,  2,  (0,0,0),  2)

    else:
        img = detector.overlayPNG(img, img_ball, [ball_pos[0], ball_pos[1]])
        ball_pos[0] += speedx
        ball_pos[1] += speedy
        if ball_pos[0] <= 20 or ball_pos[0]>1180:
            speedx = -speedx
        if ball_pos[1] <=10:
            speedy = -speedy
        if len(hands)!=0:
            for hand in hands:
                id, x, y = hand['lm_list'][8]
                x= np.clip(x, 80, 1180)
                x = smoothness * round(x/smoothness)
                h, w, c = img_bat1.shape
                x1 = x-w//2
                if hand['type']=='Left' or hand['type']=='Right':    
                    img = detector.overlay_3_channel(img, img_bat1, pos = [x1,520])
                    if 480< ball_pos[1] < 480 + h and x1-30< ball_pos[0] <x1+w+10:
                        speedy = -speedy
                        ball_pos[1] -= 30
                        score +=1
              
        cv2.putText(img,  f'Score: {score}', (1000,680),  cv2.FONT_HERSHEY_SIMPLEX,  1,  (255,255,255),  2)
        
    cv2.imshow('img',img)
    k=cv2.waitKey(1)
    if k==ord('r'):
        switch =True
        game_over =False
        ball_pos = [50,50]
        speedx = 15
        speedy = 15
        score = 0
        img_game_over = cv2.imread('Pong Game 2\\game_over.jpg')

    if k==27:
        break

cv2.destroyAllWindows()