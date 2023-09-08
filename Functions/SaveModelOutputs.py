import shutil
import os
import cv2
import numpy as np
from PIL import Image
from google.colab.patches import cv2_imshow

def modeloutputs(inputs, savename, whatmodel):
    #modeloutputs(df, 'backbonepredict_dataframe', 0)과 같이 호출하시면 됩니다.
    model = ['ForDetectron2', 'ForYolo']
    output_folderpath = '/content/drive/MyDrive/CJ대한통운 미래기술챌린지 2023/OutputModel'
    if not os.path.exists(output_folderpath):
      # os.makedirs 메소드를 사용하여 폴더를 생성
      os.makedirs(output_folderpath)
    totalpath = os.path.join(output_folderpath, os.path.join(model[whatmodel], savename))
    # csv 파일로 저장하기
    inputs.to_csv(totalpath, index=False, encoding='utf8') # index=False로 하면 인덱스를 제외하고 저장할 수 있습니다.
def copymodelpth(in_, savename, whatmodel):
    #copymodelpth(os.path.join(output_folderpath, 'ForDetectron2/detectron2model.pth'), 'detectron2model', 0)와 같이 호출하시면 됩니다.
    model = ['ForDetectron2', 'ForYolo']
    #detectron2는 pth형태로 저장하였습니다.
    #yolo는 형식에 맞게 두번째 확장자를 변경 후 저장해주세요.
    out = ['pth', 'pt']
    output_folderpath='/content/drive/MyDrive/CJ대한통운 미래기술챌린지 2023/OutputModel'
    
    shutil.copyfile(in_, os.path.join(output_folderpath, f'{model[whatmodel]}/{savename}.{out[whatmodel]}'))
#BBOX 저장 함수
def showresult(txt):
  txtlist = []
  for i, poly in enumerate(txt):
    data_list = list(poly.items()) # data_list를 list로 변환
    fields = data_list[0][1] # 인덱싱을 사용하여 fields 객체를 가져옴
    txtlist.append(fields.pred_boxes.tensor.tolist())
  return txtlist
#Percent 저장 함수
def showpercentresult(txt):
  txtlist = []
  for i, poly in enumerate(txt):
    data_list = list(poly.items()) # data_list를 list로 변환
    fields = data_list[0][1] # 인덱싱을 사용하여 fields 객체를 가져옴
    txtlist.append(fields.scores.tolist())
  return txtlist
#예측 이미지 저장 함수
#def addresultimg(list):
#  for img in list:
#    #cv2_imshow(img.get_image()[:, :, ::-1])
#    cv2_imshow(img)
def addresultimg(lst, path):
  img = cv2.imread(path)
  for idx in range(len(lst[1])):
    # 4개의 점의 좌표를 정합니다
    if lst[0][idx] == False:
      iou1_x1, iou1_y1, iou1_x2, iou1_y2 = lst[1][idx][0], lst[1][idx][1], lst[1][idx][2], lst[1][idx][3]
      p1 = (int(iou1_x1), int(iou1_y1))
      p3 = (int(iou1_x2), int(iou1_y2))
      cv2.rectangle(img, p1, p3, (255, 0, 0), 2)
  # img의 스케일을 0.5로 줄입니다.
  height, width = img.shape[:2]
  resized_img = cv2.resize(img, (int(width*0.5), int(height*0.5)))
  cv2_imshow(resized_img)
# Crop 이미지 로드 함수
def load_image(image_path):
  return cv2.imread(image_path)
# Crop 이미지 저장 함수
def save_image(image, file_name):
  if not os.path.exists("/content/drive/MyDrive/CJ대한통운 미래기술챌린지 2023/Result"):
    os.makedirs("/content/drive/MyDrive/CJ대한통운 미래기술챌린지 2023/Result")
  Image.fromarray(image).save(file_name)
def makecropimg(lst, path):
  imagefolderpath='/content/drive/MyDrive/CJ대한통운 미래기술챌린지 2023/상품매핑PreprocessedFolder'
  img = cv2.imread(path)
  for idx in range(len(lst[1])):
    # 4개의 점의 좌표를 정합니다
    iou1_x1, iou1_y1, iou1_x2, iou1_y2 = lst[1][idx][0], lst[1][idx][1], lst[1][idx][2], lst[1][idx][3]
    p1 = (int(iou1_x1), int(iou1_y1))
    p3 = (int(iou1_x2), int(iou1_y2))
    # 이미지를 잘라냅니다.
    #cropped_img = cv2.crop(img, p1, p3)
    cropped_img = img[int(iou1_y1):int(iou1_y2), int(iou1_x1):int(iou1_x2)]
    save_path = f'{imagefolderpath}/cropped_box_{1}_{idx + 1}.png'

    save_image(cropped_img, save_path)
  
