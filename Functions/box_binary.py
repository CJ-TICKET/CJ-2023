import cv2
import numpy as np
import os
import json

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

def json_folder_extension(folder_path):
    extension = '.json'

    image_paths = []
    file_names = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(extension):
            image_path = os.path.join(folder_path, file_name)
            image_paths.append(image_path)
            file_names.append(file_name)
    return image_paths, file_names


def color_to_black(img, x_list, y_list):

    #x_list = [3, 429, 433, 37] #예시임 x_list input 들어와야 하는 형태
    #y_list = [255, 277, 619, 583]
    h, w = img.shape[:2]
    img = np.zeros((h, w), np.uint8)  # 그레이스케일 이미지 생성
    points = np.array(list(zip(x_list, y_list)), np.int32)
    points = points.reshape((-1, 1, 2))
    cv2.fillPoly(img, pts= [points], color=(255, 255, 255))
    
    cv2.namedWindow('show', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('show', w//4, h//4)
    cv2.imshow('show', img)
    cv2.waitKey(0)

    return img

def validation__binary_working() :
    folder_path = '2023_cj/Dataset/validation/set1'  #validation dataset이 들어있는 폴더 경로
    json_folder_path = '2023_cj/Dataset' #json파일 폴더 경로
    image_paths, file_names = folder_extension(folder_path)
    json_paths, file_names = json_folder_extension(json_folder_path)

    for cnt, image_path in enumerate(image_paths) :
        img = cv2.imread(image_path)

        with open(json_paths[cnt], encoding='UTF8') as f :
            datas = json.load(f)
            x_list = datas['all_points_x']
            y_list = datas['all_points_y']
            img = color_to_black(img, x_list, y_list)

            ccnt = str(cnt)
            cv2.imwrite('저장 위치' + ccnt + file_names[cnt], img); cv2.waitKey(0)

validation__binary_working()

    

