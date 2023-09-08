import cv2
import numpy as np
import os

def folder_extension(folder_path):
    extension = '.jpg'

    image_paths = []
    file_names = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(extension):
            image_path = os.path.join(folder_path, file_name)
            image_paths.append(image_path)
            file_names.append(file_name)
    return image_paths, file_names

def image_hsv(image) :
    copy = image.copy()
    copy = cv2.resize(copy, dsize = (0, 0), fx = 0.3, fy = 0.3, interpolation = cv2.INTER_AREA)
    hsv = cv2.cvtColor(copy, cv2.COLOR_BGR2HSV)
    # green
    lower_green1 = np.array([30, 25, 25])
    upper_green1 = np.array([0, 255, 255])
    mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
    lower_green2 = np.array([30, 25, 25])
    upper_green2 = np.array([80, 255, 255])
    mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
    mask = mask1+mask2

    cv2.imshow("green", mask); cv2.waitKey(0)

    return mask



folder_path = ''
image_paths, file_names = folder_extension(folder_path)

for cnt, image_path in enumerate(image_paths) :
    img = cv2.imread(image_path)
    img = image_hsv(img)
    #cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    #cv2.resizeWindow(win_name, cols//4, rows//4)
    #cv2.imshow(win_name, img)
    #cv2.waitKey(0)

