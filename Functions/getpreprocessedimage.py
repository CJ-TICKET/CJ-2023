import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import rembg
import os.path
import gc
import psutil

def getnukki(image):
    image = rembg.remove(image)

    return image

def showimagesinarow(nukki=False, dataset=None, trainortest=None):
  start = time.time()
  imagelist = []
  labels = []
  polyimagelist = []
  width = 4
  height = 10
  data = []
  for filename in os.listdir('/content/drive/MyDrive/CJ대한통운 미래기술챌린지 2023/상품매핑TrainTestFolder/train'):
    filepath = os.path.join('/content/drive/MyDrive/CJ대한통운 미래기술챌린지 2023/상품매핑TrainTestFolder/train', filename)
    filepath = filepath.split('/train/')[1]
    label = filename.split(')')[0]
    data.append({'Filepath': filepath, 'label': label})
  df = pd.DataFrame(data)

  if dataset == None :
    fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(width * 7, height))
    for label in range(6):
      imagelist.append(np.array(Image.open(
          os.path.join('/content/drive/MyDrive/CJ대한통운 미래기술챌린지 2023/상품매핑TrainTestFolder/train', df['Filepath'][0 + 20 * label]))))
      labels.append(df['label'][0 + 20 * label])

    pbar = tqdm(imagelist,
              total = 6, ## 전체 진행수
              desc = 'Show image progress', ## 진행률 앞쪽 출력 문장
              ncols = 100, ## 진행률 출력 폭 조절
              ascii = ' =', ## 바 모양, 첫 번째 문자는 공백이어야 작동
              leave = True, ## True 반복문 완료시 진행률 출력 남김. False 남기지 않음.
    )

    if nukki == True:
      for i, image in enumerate(pbar):
        result = getnukki(image)
        polyimagelist.append(result)


        axes[i].imshow(result)
        axes[i].axis('off')
        axes[i].set_title(labels[i])
    else :
      for i, image in enumerate(pbar):
        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].set_title(labels[i])
    print(f"\nRunning cnt: {len(imagelist)}개의 이미지 연산")
    plt.show()
  else :
    if nukki == True:
      for i in range(len(dataset[0])):
        filepath = dataset[1][i]
        label = dataset[1][i].split(f'/{trainortest}/')[1].split(')')[0]
        result = getnukki(dataset[0][i])
        image = Image.fromarray(np.uint8(result)).convert('RGB')
        image.save(filepath)
      print(f"\nRunning cnt: {len(dataset[0])}개의 이미지 연산")
  end = time.time()
  runningtime = end - start

  print(f"\nRunning Time: {runningtime:.2f}초")

def allimagepreprocessing(dataurl, trainortest):
  imagelist = []
  labellist = []
  runningfiles = os.listdir(dataurl)
  file_names = [os.path.join(dataurl, file) for file in runningfiles]
  pbar = tqdm(file_names,
              total = len(file_names), ## 전체 진행수
              desc = 'Show image progress', ## 진행률 앞쪽 출력 문장
              ncols = 100, ## 진행률 출력 폭 조절
              ascii = ' =', ## 바 모양, 첫 번째 문자는 공백이어야 작동
              leave = True, ## True 반복문 완료시 진행률 출력 남김. False 남기지 않음.
    )
  for i, filename in enumerate(pbar):
    img = cv2.imread(filename)
    imagelist.append(np.array(img))
    labellist.append(filename)
    del img # 이미지 메모리 해제
    gc.collect()
    time.sleep(0.1)
    if psutil.virtual_memory().percent > 70:
      print(f"\nRam 용량 초과로 일부의 이미지를 미리 연산 중입니다!")
      #주석처리한 아래 코드는 nukki로 배경을 없애줍니다.
      showimagesinarow(nukki=True, dataset=[imagelist, labellist], trainortest=trainortest)
      del imagelist
      del labellist
      gc.collect()
      imagelist = []
      labellist = []
      print(f"\nRam 용량 초과로 일부의 이미지를 연산하였으며 남은 이미지를 처리중입니다! 잠시만 기다려주세요!")
  #주석처리한 아래 코드는 nukki로 배경을 없애줍니다.
  showimagesinarow(nukki=True, dataset=[imagelist, labellist], trainortest=trainortest)
  print(f"\nRunning cnt: {len(runningfiles)}개의 이미지 연산완료!")
def imagebyimage(dataurl):
  runningfiles = os.listdir(dataurl)
  file_names = [os.path.join(dataurl, file) for file in runningfiles]
  pbar = tqdm(file_names,
              total = len(file_names), ## 전체 진행수
              desc = 'Show image progress', ## 진행률 앞쪽 출력 문장
              ncols = 100, ## 진행률 출력 폭 조절
              ascii = ' =', ## 바 모양, 첫 번째 문자는 공백이어야 작동
              leave = True, ## True 반복문 완료시 진행률 출력 남김. False 남기지 않음.
    )
  for i, filename in enumerate(pbar):
    img = cv2.imread(filename)
    filepath = filename.replace('image_preprocessing/NewImage', '상품매핑TrainTestFolder/train')
    result = getnukki(np.array(img))
    image = Image.fromarray(np.uint8(result)).convert('RGB')
    image.save(filepath)
  print(f"\nRunning cnt: {len(runningfiles)}개의 이미지 연산완료!")