import os
import cv2
import random

# 디렉터리 경로 설정
input_directory = 'C:/Users/dpffl/Downloads/2023_cj/total_img'  # augmentation을 적용할 원본 이미지가 있는 디렉터리
output_directory = 'C:/Users/dpffl/Downloads/2023_cj/aug_line2'  # augmentation된 이미지를 저장할 디렉터리

# 디렉터리 내의 모든 이미지 파일 읽기
for filename in os.listdir(input_directory):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        input_image_path = os.path.join(input_directory, filename)
        output_image_path = os.path.join(output_directory, filename)

        # 이미지 로드
        image = cv2.imread(input_image_path)
        height, width, _ = image.shape

        crop_size = min(height, width)
        start_x = (width - crop_size) // 2
        start_y = (height - crop_size) // 2
        
        #####(1)이미지 line
        line_thickness = random.randint(30, 35)

        # 랜덤한 선의 시작점과 끝점 생성
        start_point = (0, height//2)
        end_point = (width-1, height//2)

        # 이미지에 선 그리기
        cv2.line(image, start_point, end_point, (0, 255, 0), line_thickness)

        #####(2)이미지  blur
        #blurred_image = cv2.GaussianBlur(image, (3, 3), 0)

        #####(3)이미지 flip
        #flipped_image = cv2.flip(image, 1)

        ######(4)이미지 crop
        #cropped_image = image[start_y:start_y + crop_size, start_x:start_x + crop_size]


        ######(5) 랜덤 회전 적용
        #angle = random.randint(0, 180)
        #rows, cols, _ = image.shape
        #rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        #rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))

        # 이미지 저장
        cv2.imwrite(output_image_path, image)

print("Image augmentation completed for all images.")