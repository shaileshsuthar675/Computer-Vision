import cv2
import os
import random

img = random.choice(os.listdir("collage center"))
overlay_img = cv2.imread('output/1.jpg')
overlay_img = cv2.resize(overlay_img, (1960, 1080))
v_stack_img = cv2.imread('output/collage_img.png')

final_img = cv2.addWeighted(v_stack_img, 0.6, overlay_img, 0.7, 0)
cv2.imshow('final_img', final_img)
cv2.imwrite('output/final_img.png', final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
