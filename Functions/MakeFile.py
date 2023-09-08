import numpy as np
import cv2
import os
import json
import shutil
from detectron2.structures import BoxMode

# 연습 시에 set10을 따로 test로 나누기 위해 작성한 함수입니다.
# validation폴더에서 set10만 따로 아래에 정의한 경로로 나눕니다.
# 예선 시에는 필요하지 않습니다.
def splittotraintestfolder():
    original_path="/content/drive/MyDrive/CJ대한통운 미래기술챌린지 2023"
    validataion_path = os.path.join(original_path, 'validation')
    dst_path_test='/content/drive/MyDrive/CJ대한통운 미래기술챌린지 2023/Validation_TrainTest/Test'
    dst_path_train='/content/drive/MyDrive/CJ대한통운 미래기술챌린지 2023/Validation_TrainTest/Train'

    if not os.path.exists(dst_path_train):
        for var in os.listdir(validataion_path):
          if var == 'set10':
            shutil.copytree(os.path.join(validataion_path, var), dst_path_test)
          else:
            shutil.copytree(os.path.join(validataion_path, var), os.path.join(dst_path_train, var))
        for folder in os.listdir(dst_path_train):
            for filename in os.listdir(os.path.join(dst_path_train, folder)):
                shutil.move(os.path.join(dst_path_train, folder, filename), dst_path_train)
        for var in os.listdir(validataion_path):
          if var == 'set10':
            None
          else:
            os.rmdir(os.path.join(dst_path_train, var))
    else:
        for folder in os.listdir(dst_path_train):
            if os.path.isdir(folder):
                for filename in os.listdir(os.path.join(dst_path_train, folder)):
                    shutil.move(os.path.join(dst_path_train, folder, filename), dst_path_train)
        for var in os.listdir(validataion_path):
          if var == 'set10':
            None
          else:
            if not os.path.exists(os.path.join(dst_path_train, var)):
              None
            else:
              os.rmdir(os.path.join(dst_path_train, var))
              
def splitjson(num=99, isnewfile=False, filename=None):
    if not os.path.exists('/content/drive/MyDrive/CJ대한통운 미래기술챌린지 2023/Json'):
      os.makedirs('/content/drive/MyDrive/CJ대한통운 미래기술챌린지 2023/Json')
    if isnewfile == True:
      json_path=filename
    else:
      json_path='/content/drive/MyDrive/CJ대한통운 미래기술챌린지 2023/Json/ForDetectron2/validation1_10.json'

    with open(json_path) as f:
        data = json.load(f)
    # index를 저장할 변수 초기화
    index = -1
    if type(data) == dict:
      data_list = list(data.items())
    else:
      data_list = data
    for i, d in enumerate(data_list): # enumerate를 사용하여 인덱스와 값을 함께 얻음
        if d[0].startswith('cart_'): # d[0]은 키, d[1]은 값
            number = int(d[0].split('.')[0].split('_')[1][:2])
            if number == num: # number와 num이 같은지 확인
                index = i # 인덱스를 저장하고
                break # 반복문을 종료
    # index가 -1이면 number와 num이 같은 값이 없다는 뜻
    if index == -1:
        #print("No matching value found")
        data_1 = data_list[:] # 처음부터 끝까지
        return data_1
    else: # index가 -1이 아니면 슬라이싱을 사용하여 데이터를 나눔
        data_1 = data_list[:index] # 처음부터 index-1까지
        data_2 = data_list[index:] # index부터 끝까지
        return data_1, data_2 # 나눈 데이터를 반환
    
def makejsonfile(jsondict, savename, whatmodel):
    model = ['ForDetectron2', 'ForYolo']
    original_path="/content/drive/MyDrive/CJ대한통운 미래기술챌린지 2023"
    # json 파일로 저장할 경로와 이름 지정
    json_folderpath = os.path.join(original_path, f'Json/{model[whatmodel]}')
    json_path = os.path.join(json_folderpath, f'{savename}.json')

    # 파일을 쓰기 모드로 열고
    if not os.path.exists(json_folderpath):
        # os.makedirs 메소드를 사용하여 폴더를 생성
        os.makedirs(json_folderpath)
    with open(json_path, "w") as f:
        # json.dump 메소드를 사용하여 딕셔너리를 json 파일로 저장
        json.dump(jsondict, f, ensure_ascii=False)

        
# detectron2에서 이미지의 정보를 수집하는 함수를 아래에 작성하였습니다.
# 라벨은 한글이 깨지고 있어 boxes로 통일합니다.
def get_set_number(filename):
    if filename.startswith('cart_'):
      count = str(int(filename.split('.')[0].split('_')[1][:2]))
      num = f'set{count}'
      return num
def get_dicts(img_dir, jsonfilename):
    listofproducts = []
    json_file = os.path.join(jsonfilename)
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for i in range(len(imgs_anns)):
        for idx, v in enumerate(imgs_anns[i]):
            if idx == 1 :
                record = {}
                new_name = get_set_number(v["filename"])
                
                if new_name == 'set11':
                  filename = os.path.join(os.path.join('/content/drive/MyDrive/CJ대한통운 미래기술챌린지 2023/NewImage', "testimage"), v["filename"])
                else:
                  filename = os.path.join(os.path.join(os.path.join(img_dir, "validation"), new_name), v["filename"])
                height, width = cv2.imread(filename).shape[:2]

                record["file_name"] = filename
                record["image_id"] = idx
                record["height"] = height
                record["width"] = width

                annos = v["regions"]
                objs = []
                for i, anno in enumerate(annos):
                    #assert not anno["region_attributes"]
                    #value = anno["region_attributes"]['상품']
                    #listofproducts.append(value)
                    #list(set(listofproducts))
                    anno_val = anno["shape_attributes"]
                    px = anno_val["all_points_x"]
                    py = anno_val["all_points_y"]
                    poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                    poly = [p for x in poly for p in x]

                    obj = {
                        "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": [poly],
                        "category_id": 0,
                    }
                    objs.append(obj)
                record["annotations"] = objs
                dataset_dicts.append(record)
    return dataset_dicts