#생성한 ComUtils에 정의된 함수를 사용하기 위해 사용
from util.CommUtils import *

def preprocessing():

    image = cv2.imread("image/minji.jpg", cv2.IMREAD_COLOR)

    if image is None: return None, None

    image = cv2.resize(image, (600, 800))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray =cv2.equalizeHist(gray)

    return image, gray

face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

eye_cascade = cv2.CascadeClassifier("data/haarcascade_eye_tree_eyeglasses.xml")

#인식률을 높이기 위한 전처리 함수 호출
image, gray = preprocessing()   # 전처리

if image is None: raise Exception("영상 파일 읽기 에러")

faces = face_cascade.detectMultiScale(gray, 1.1, 2, 0, (100, 100))

# 얼굴이 검출되었다면
if faces.any():

    # 얼굴 위치 값을 가져오기
    x, y, w, h = faces[0]

    # 원본이미지로부터 얼굴영역 가져오기
    face_image = image[y:y + h, x:x + w]

    eyes = eye_cascade.detectMultiScale(face_image, 1.15, 7, 0, (25, 20))

    # 눈을 찾을 수 있다면,
    if len(eyes) == 2:

        # 얼굴 가운데
        face_center = (int(x + w // 2), int(y + h // 2))

        # 양쪽 눈 가운데 위치 값 가져오기
        eye_centers = [[x+ex+ew//2, y+ey+eh//2] for ex,ey,ew, eh in eyes]

        # 사진의 기울기
        correction_image, correction_center = doCorrectionImage(image, face_center, eye_centers )

        # 얼굴 상세 객체(윗머리, 귀밑머리, 입술) 찾기
        rois = doDetectObject(faces[0], face_center)

        # 얼굴 검출 사각형 그리기
        cv2.rectangle(correction_image, rois[0], (255, 0, 255), 2)  # 윗머리 영역
        cv2.rectangle(correction_image, rois[1], (255, 0, 255), 2)  # 귀밑머리 영역

        # 입술 검출 사각형 그리기
        cv2.rectangle(correction_image, rois[2], (255, 0, 0), 2)

        cv2.circle(correction_image, tuple(correction_center[0]), 5, (0, 255, 0), 2)
        cv2.circle(correction_image, tuple(correction_center[1]), 5, (0, 255, 0), 2)

        cv2.circle(correction_image, face_center, 3, (0, 0, 255), 2)

        cv2.imshow("MyFace_Edit", correction_image)

    else:
        print("눈 미검출")

else:
    print("얼굴 미검출")

cv2.waitKey(0)