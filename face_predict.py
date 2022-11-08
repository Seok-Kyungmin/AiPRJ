import cv2

face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

vcp = cv2.VideoCapture(0, cv2.CAP_DSHOW)

model = cv2.face.LBPHFaceRecognizer_create()

# 학습된 모델 가져오기
model.read("model/face-trainner.yml") # 저장된 값 가져오기

# 카메라로부터 이미지 가져오기
while True:
    ret, my_image = vcp.read()

    # 동영상으로부터 프레임(이미지)를 잘 받았으면 실행함
    if ret is True:

        gray = cv2.cvtColor(my_image, cv2.COLOR_BGR2GRAY)

        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(gray, 1.5, 5, 0, (20, 20))

        facesCnt = len(faces)

        if facesCnt == 1:

            x, y, w, h = faces[0]

            face_image = gray[y:y + h, x:x +w]

            # 유사도 분석
            id_, res = model.predict(face_image)

            # 에측결과 문자열
            result = "result : " + str(res) + "%"

            # 예측결과 문자열 사진에 추가하기
            cv2.putText(my_image, result, (x, y - 15), 0, 1, (255, 0, 0), 2)

            # 얼굴 검출 사각형 그리기
            cv2.rectangle(my_image, faces[0], (255, 0, 0), 4)

        # 사이즈 변경된 이미지로 출력하기
        cv2.imshow("predict_my_face", my_image)

    # 입력받는 것 대기하기, 작성안하면, 결과창이 바로 닫힘
    if cv2.waitKey(1) > 0:
        break

vcp.release()

cv2.destroyAllWindows()