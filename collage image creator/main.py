import numpy as np
import cv2
import os
import random

img = random.choice(os.listdir("collage center"))
overlay_img = cv2.imread('output/1.jpg')
overlay_img = cv2.resize(overlay_img, (1960, 1080))
img_list = os.listdir("collage center")


# creating a function that helps to select the random image base on the quantity that we needed for making collage image
def img_randomness(img_name_list, length=15):
    img_list_updated = []
    for _ in range(length):
        img_list_updated.append(img_name_list[random.randint(0, len(img_name_list) - 1)])
    return img_list_updated


v_stack_img_list = []
img_stack_length = 40
for j in range(img_stack_length):
    img_list_l = img_randomness(img_list, img_stack_length)
    print(j)
    h_stack_img_list = []
    for i in img_list_l:
        img = cv2.imread(f'collage center/{i}')
        img = cv2.resize(img, (int(1980 / img_stack_length), int(1080 / img_stack_length)),
                         interpolation=cv2.INTER_AREA)
        h_stack_img_list.append(img)
    v_stack_img_list.append(np.hstack(h_stack_img_list))

v_stack_img = np.vstack(v_stack_img_list)
cv2.imshow('v_stack_img', v_stack_img)
cv2.imwrite('output/collage_img.png', v_stack_img)
final_img = cv2.addWeighted(v_stack_img, 0.6, overlay_img, 0.7, 0)
cv2.imshow('final_img', final_img)
cv2.imwrite('output/final_img.png', final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
