import cv2
import faces as faces

image = cv2.imread("image/bp.jpg", cv2.IMREAD_COLOR)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.equalizeHist(gray)

if image is None: raise Exception("이미지 읽기 실패")

face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

faces = face_cascade.detectMultiScale(gray, 1.1, 5, 0, (100,100))

facesCnt = len(faces)

print(len(faces))

if facesCnt > 0:

    for face in faces:

        x, y, w, h = face

        cv2.rectangle(image, face, (255, 0, 0), 4)

else: print("얼굴 미검출")

cv2.imshow("MyFace", image)

cv2.waitKey(0)
