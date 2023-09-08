import cv2
import numpy as np
import os
import matplotlib.pyplot as plt 

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

def onMouse(event, x, y, flags, param):
    #event : 윈도우에서 발생하는 이벤트
    #x, y 마우스의 좌표
    #flags는 event와 함께 활용되는 역할로 특수한 상태를 확인하는 용도
    #params는 마우스 콜백 설정 함수(cv2.setMouseCallback)에서 함께 전달되는 사용자 정의 데이터
    
    h, w = draw.shape[:2]
    global pts_cnt
    if event == cv2.EVENT_LBUTTONDOWN:  #마우스 왼쪽 버튼을 누를 때

        # 좌표에 초록색 동그라미 표시
        cv2.circle(draw, (x, y), 10, (0, 255, 0), -1)
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, w//4, h//4)
        cv2.imshow(win_name, draw)

        # 마우스 좌표 저장
        pts[pts_cnt] = [x, y]
        pts_cnt += 1
        if pts_cnt == 4:
            ### 좌표 4개 중 상하좌우 찾기
            #sm = pts.sum(axis=1)  # 4쌍의 좌표 각각 x+y 계산
            #diff = np.diff(pts, axis=1)  # 4쌍의 좌표 각각 x-y 계산

            #topLeft = pts[np.argmin(sm)]  # x+y가 가장 값이 좌상단 좌표
            #bottomRight = pts[np.argmax(sm)]  # x+y가 가장 큰 값이 우하단 좌표
            #topRight = pts[np.argmin(diff)]  # x-y가 가장 작은 것이 우상단 좌표
            #bottomLeft = pts[np.argmax(diff)]  # x-y가 가장 큰 값이 좌하단 좌표

            #순서를 직접 지정 
            topLeft = pts[0]
            bottomLeft = pts[1]
            bottomRight = pts[2]
            topRight = pts[3]

            # 변환 전 4개 좌표 
            pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])
            
            # 변환 후 영상에 사용할 서류의 폭과 높이 계산
            w1 = abs(bottomRight[0] - bottomLeft[0])
            w2 = abs(topRight[0] - topLeft[0])
            h1 = abs(topRight[1] - bottomRight[1])
            h2 = abs(topLeft[1] - bottomLeft[1])
            width = int(max([w1, w2]))  # 두 좌우 거리간의 최대값이 서류의 폭
            height = int(max([h1, h2]))  # 두 상하 거리간의 최대값이 서류의 높이

            print(w1, w2, h1, h2)

            # 변환 후 4개 좌표
            pts2 = np.float32([[0, 0], [width - 1, 0],
                               [width - 1, height - 1], [0, height - 1]])

            # 변환 행렬 계산 
            mtrx = cv2.getPerspectiveTransform(pts1, pts2)
            # 원근 변환 적용
            result = cv2.warpPerspective(img, mtrx, (width, height))
            cv2.namedWindow('scanned', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('scanned', w//4, h//4)
            cv2.imshow('scanned', result); cv2.waitKey(0)
            cv2.imwrite('C:/Users/dpffl/Downloads/2023_cj/image2/' + param + '.jpg',result); cv2.waitKey(0)

            return result

folder_path = 'C:/Users/dpffl/Downloads/2023_cj/Dataset/image1'
image_paths, file_names = folder_extension(folder_path)

for cnt, image_path in enumerate(image_paths) :
    img = cv2.imread(image_path)
    rows, cols = img.shape[:2]
    draw = img.copy()
    pts_cnt = 0
    win_name = "scanning"
    pts = np.zeros((4, 2), dtype=np.float32)
    
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, cols//4, rows//4)
    cv2.imshow(win_name, img)
    cv2.setMouseCallback(win_name, onMouse, param = file_names[cnt])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

        

