import os.path
import random
from tqdm import tqdm
import shutil
import requests
from urllib.parse import urlparse
import zipfile
from PIL import Image

def putindata(getdir, imgurl, imgname):
    image_url = imgurl
    # 이미지 데이터를 받아옵니다.
    img_data = requests.get(image_url).content
    if img_name == None:
        img_name = urlparse(image_url) # URL을 구성 요소별로 분리합니다.
        file_name = img_name.path.split('/')[-1]
    else:
        file_name = imgname
    # 임시로 이미지 파일을 생성합니다.
    with open(file_name, 'wb') as handler:
        handler.write(img_data)
    # 구글 드라이브 폴더에 이미지 파일을 복사합니다.
    shutil.copy(file_name, getdir)

    # 임시 파일을 삭제합니다.
    os.remove(file_name)
    
def split_data(src_dir, train_dir, test_dir, ratio):
    #train : "/content/drive/MyDrive/CJ대한통운 미래기술챌린지 2023/상품매핑TrainTestFolder/train" 폴더 생성
    #test : "/content/drive/MyDrive/CJ대한통운 미래기술챌린지 2023/상품매핑TrainTestFolder/test" 폴더 생성
    make_dir(train_dir, test_dir)
    doingsplitaction(src_dir, train_dir, test_dir, ratio)
        
          
def make_dir(train_dir, test_dir):
    if os.path.exists(train_dir):
        None
    else:
        os.makedirs(train_dir)
        os.makedirs(test_dir)
        
def doingsplitaction(src_dir, train_dir, test_dir, ratio):
    #src_dir의 모든 이미지파일을 읽어서 shuffle
    files = os.listdir(src_dir)
    random.shuffle(files)
    train_files = files[:int(len(files) * ratio)]
    test_files = files[int(len(files) * ratio):]
    print('Train : {}'.format(len(train_files)))
    print('Test : {}'.format(len(test_files)))
    pbar1 = tqdm(train_files,
        total = len(train_files), ## 전체 진행수
        desc = 'Description Shutil TrainFolder', ## 진행률 앞쪽 출력 문장
        ncols = 100, ## 진행률 출력 폭 조절
        ascii = ' =', ## 바 모양, 첫 번째 문자는 공백이어야 작동
        leave = True, ## True 반복문 완료시 진행률 출력 남김. False 남기지 않음.
    )
    for file in pbar1:
        src_file = os.path.join(src_dir, file)
        dst_file = os.path.join(train_dir, file)
        shutil.copyfile(src_file, dst_file)
    pbar2 = tqdm(test_files,
        total = len(test_files), ## 전체 진행수
        desc = 'Description Shutil TestFolder', ## 진행률 앞쪽 출력 문장
        ncols = 100, ## 진행률 출력 폭 조절
        ascii = ' =', ## 바 모양, 첫 번째 문자는 공백이어야 작동
        leave = True, ## True 반복문 완료시 진행률 출력 남김. False 남기지 않음.
    )
    for file in pbar2:
        src_file = os.path.join(src_dir, file)
        dst_file = os.path.join(test_dir, file)
        shutil.copyfile(src_file, dst_file)

def loadinzip(src_dir):
    # zip폴더 내부의 모든 폴더 경로를 지정합니다
    # 이후 split_data 함수를 호출하여 ratio에 맞게 폴더의 이미지들을 나누어 TrainTestFolder로 넣는 로직을 사용합니다.
    # src_dir = "/content/drive/MyDrive/CJ대한통운 미래기술챌린지 2023/augmentation"
    folders = os.listdir(src_dir)
    folder_path = [os.path.join(src_dir, folder) for folder in folders]
    return folder_path

def removeinFolder(src_dir):
    for files in os.listdir(src_dir):
        os.remove(os.path.join(src_dir, files))