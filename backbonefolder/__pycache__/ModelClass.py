import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm
import time
import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3, EfficientNetB4

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = models.resnet50(weights='IMAGENET1K_V2')
        self.resnet.fc = nn.Linear(2048, 1000)

    def forward(self, x):
        x = self.resnet(x)
        return x
    def summary(self):
        s = self.resnet
        return s
    
class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.googlenet = models.googlenet(weights='IMAGENET1K_V1')
        self.googlenet.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.googlenet(x)
        return x

    def summary(self):
        s = self.googlenet
        return s
    
class ResGoogLeNet(nn.Module):
    def __init__(self):
        super(ResGoogLeNet, self).__init__()
        self.resnet = models.resnet50(weights='IMAGENET1K_V2')
        self.googlenet = models.googlenet(weights='IMAGENET1K_V1')
        self.fc = nn.Linear(2048, 1024)
        self.resnet.fc = nn.Sequential(self.fc, self.googlenet.fc)
        self.out_channels = 2048

    def forward(self, x):
        x = self.resnet(x)
        return x

class EfficientNetB4(nn.Module):
    def __init__(self):
        super(EfficientNetB4, self).__init__()
        self.efficientnetb4 = EfficientNetB4(weights='imagenet') # 수정된 부분
        self.fc = nn.Linear(1792, 1000)

    def forward(self, x):
        x = self.efficientnetb4(x)
        x = self.fc(x) # 수정된 부분
        return x

    def summary(self):
        s = self.efficientnetb4
        return s
    
class MyDataset(Dataset):
    def __init__(self, labeledimg_dir):
        self.labeledimg_dir = labeledimg_dir
        self.imgs = os.listdir(labeledimg_dir)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.labeledimg_dir, self.imgs[idx])
        img = Image.open(img_path).convert('RGB')
        #augmentation이후 수정된 코드입니다.
        #label = int(self.imgs[idx].split(')')[0])
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img = transform(img)
        if self.imgs[idx].split('_')[1][:2 if len(self.imgs[idx].split('_')[1]) == 4 else 3].isdigit(): # 문자열이 숫자로만 이루어져 있는지 확인합니다.
            num = int(self.imgs[idx].split('_')[1][:2 if len(self.imgs[idx].split('_')[1]) == 4 else 3]) # 문자열을 정수로 변환합니다.
            label = torch.tensor(num)
            return img, label
        else: # 그렇지 않으면
            return img
    
def train(model, dataloader, criterion, optimizer):
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    start = time.time()
    model.train() # 모델을 학습 모드로 설정합니다.
    correct_train = 0 # 정답 개수를 초기화합니다.
    total_train = 0 # 전체 개수를 초기화합니다.
    pbar = tqdm(dataloader,
              total = len(dataloader), ## 전체 진행수
              desc = 'Description Training', ## 진행률 앞쪽 출력 문장
              ncols = 100, ## 진행률 출력 폭 조절
              ascii = ' =', ## 바 모양, 첫 번째 문자는 공백이어야 작동
              leave = True, ## True 반복문 완료시 진행률 출력 남김. False 남기지 않음.
    )
    for imgs, labels in pbar: # dataloader에서 이미지와 라벨을 반복적으로 불러옵니다.
        imgs, labels = imgs.to(device), labels.to(device) # 이미지와 라벨을 해당 장치로 옮깁니다.
        outputs = model(imgs) # 모델에 이미지를 입력하여 출력값을 얻습니다.
        loss = criterion(outputs, labels) # 손실 값을 계산합니다.
        optimizer.zero_grad() # 최적화 함수의 기울기를 초기화합니다.
        loss.backward() # 손실 값에 대한 기울기를 계산합니다.
        optimizer.step() # 모델의 가중치를 업데이트합니다.
        _, predicted_train = torch.max(outputs.data, 1) # 출력값에서 가장 큰 값의 인덱스를 예측값으로 사용합니다.
        correct_train += torch.sum(torch.eq(labels, predicted_train)).item() # 예측값과 라벨값이 일치하는 개수를 세고, correct_train에 더합니다.
        total_train += labels.size(0) # 라벨값의 개수를 세고, total_train에 더합니다.
    end = time.time()
    runningtime = end - start
    return runningtime, correct_train / total_train # 정확도 값을 반환합니다.
def test(model, dataloader):
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    start = time.time()
    model.eval() # 모델을 평가 모드로 설정합니다.
    accuracy = 0 # 정확도 값을 초기화합니다.
    pbar = tqdm(dataloader,
              total = len(dataloader), ## 전체 진행수
              desc = 'Description Testing', ## 진행률 앞쪽 출력 문장
              ncols = 100, ## 진행률 출력 폭 조절
              ascii = ' =', ## 바 모양, 첫 번째 문자는 공백이어야 작동
              leave = True, ## True 반복문 완료시 진행률 출력 남김. False 남기지 않음.
             )
    with torch.no_grad(): # 가중치 업데이트와 메모리 사용량을 방지합니다.
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device) # 이미지와 라벨을 해당 장치로 옮깁니다.
            outputs = model(imgs) # 모델에 이미지를 입력하여 출력값을 얻습니다.
            _, predicted = torch.max(outputs.data, 1) # 출력값에서 가장 큰 값의 인덱스를 예측값으로 사용합니다.
            # accuracy += sklearn.metrics.accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy()) # sklearn의 함수를 사용하여 정확도 값을 갱신합니다.
            accuracy += torch.sum(torch.eq(labels, predicted)).item() / len(labels) # 직접 정확도 값을 계산하여 갱신합니다.
    end = time.time()
    runningtime = end - start
    return runningtime, accuracy / len(dataloader) # 정확도 값을 평균하여 반환합니다.
def predictfunc(model, dataloader):
    predictedlist = []
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    start = time.time()
    model.eval() # 모델을 평가 모드로 설정합니다.
    pbar = tqdm(dataloader,
              total = len(dataloader), ## 전체 진행수
              desc = 'Description Testing', ## 진행률 앞쪽 출력 문장
              ncols = 100, ## 진행률 출력 폭 조절
              ascii = ' =', ## 바 모양, 첫 번째 문자는 공백이어야 작동
              leave = True, ## True 반복문 완료시 진행률 출력 남김. False 남기지 않음.
             )
    with torch.no_grad(): # 가중치 업데이트와 메모리 사용량을 방지합니다.
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device) # 이미지와 라벨을 해당 장치로 옮깁니다.
            outputs = model(imgs) # 모델에 이미지를 입력하여 출력값을 얻습니다.
            _, predicted = torch.max(outputs.data, 1) # 출력값에서 가장 큰 값의 인덱스를 예측값으로 사용합니다.
            predictedlist.append(predicted)
    return predictedlist