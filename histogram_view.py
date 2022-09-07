import cv2
from matplotlib import pyplot as plt

image_file = "image/face.jpg"

original = cv2.imread(image_file, cv2.IMREAD_COLOR)

gray = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

unchange = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)

color = ('b', 'g', 'r')
#색상이 존재하는 원본이미지 히스토그램 보기
#BGR 순서대로 그래프 그리기 위해 반복문 사용함
for i, col in enumerate(color):

    #calHist 파라미터 설명
    # 첫번째 파라미터(image) : 분석할 이미지 파일
    # 두번째 파라미터(Channel) : 컬러이미지(BGR)이면, 배열 값 3개로 정의
    # 세번째 파라미터(Mask) : 분석할 영역의 형태인 mask
    hist = cv2.calcHist([original], [i], None, [256], [0, 256])
    plt.figure(1)
    plt.plot(hist, color = col)

plt.show()

hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.figure(2)
plt.plot(hist)
plt.show()

gray = cv2.equalizeHist(gray)

hist = cv2.calHist([gray], [0], None, [256],[0, 256])
plt.figure(3)
plt.plot(hist)
plt.show()