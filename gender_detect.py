#생성한 ComUtils에 정의된 함수를 사용하기 위해 사용
from util.CommUtils import *

def preprocessing():

    image = cv2.imread("image/mb.jpg", cv2.IMREAD_COLOR)

    if image is None: return None, None

    image = cv2.resize(image, (700, 600))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray =cv2.equalizeHist(gray)

    return image, gray

# 얼굴 탐지를 위한 모델
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
        face_center = int(x + w // 2), int(y + h // 2)

        # 양쪽 눈 가운데 위치 값 가져오기
        eye_centers = [[x+ex+ew//2, y+ey+eh//2] for ex, ey, ew, eh in eyes]

        # 사진의 기울기 보정
        correction_image, correction_center = doCorrectionImage(image, face_center, eye_centers)

        # 얼굴 상세 객체(윗머리, 귀밑머리, 입술) 찾기
        rois = doDetectObject(faces[0], face_center)

        # 보정된 사진 전체를 마스크만들기
        bace_mask = np.full(correction_image.shape[:2], 255, np.uint8)

        # 얼굴 전체 마스크 만들기(사람의 얼굴은 평균 약 45% 타원으로 구성됨)
        face_mask = draw_ellipse(bace_mask, rois[3], 0, -1)

        # 입술 마스크 만들기
        lip_mask = draw_ellipse(np.copy(bace_mask), rois[2], 255)

        masks = [face_mask, face_mask, lip_mask, ~lip_mask]

        # 만들어 놓은 마스크에 얼굴 상세 객체의 크기에 맞게 다시 저장함
        masks = [mask[y:y + h, x:x + w] for mask, (x, y, w, h) in zip(masks, rois)]

        # 얼굴 영역별 이미지 생성
        subs = [correction_image[y:y+h, x:x+w] for x, y, w, h in rois]

        hists = [cv2.calcHist([sub], [0, 1, 2], mask, (128, 128, 128), (0, 256, 0, 256, 0, 256)) for sub, mask
                 in zip(subs, masks)]

        # 각 얼굴 영역별 히스트 값의 평균 구함
        hists = [h / np.sum(h) for h in hists]

        # 얼굴색과 입술색 비교 / HISTCMP_CORREL : 1에 가까울수록 유사함
        sim1 = cv2.compareHist(hists[3], hists[2], cv2.HISTCMP_CORREL)

        # 윗머리와 귀밑머리 비교 / HISTCMP_CORREL :1에 가까울수록 유사함
        sim2 = cv2.compareHist(hists[3], hists[1], cv2.HISTCMP_CORREL)

        # 여자 남자 구별하도록 만든 공식
        # 얼굴색과 입술색이 0.2보다 크면 0.2로 정의함
        criteria = 0.2 if sim1 > 0.2 else 0.1

        # 윗머리와 귀밑머리 유사도가 얼굴색과 입술색 유사도보다 크면 생성
        value = sim2 > criteria

        # 출력 문구
        text = "Woman" if value else "Man"

        # 이미지에 표기할 문구
        cv2.putText(image, text, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # 사이즈 변경된 이미지로 출력하기
        cv2.imshow("MyFace", image)

    else:
        print("눈 미검출")

else:
    print("얼굴 미검출")

cv2.waitKey(0)