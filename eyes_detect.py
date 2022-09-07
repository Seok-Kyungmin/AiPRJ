import cv2

def preprocessing():

    image = cv2.imread("image/face.jpg", cv2.IMREAD_COLOR)

    if image is None: return None, None

    image = cv2.resize(image, (700, 700))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow("gray1", gray)

    gray = cv2.equalizeHist(gray)

    cv2.imshow("gray2", gray)

    return image, gray

face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

eye_cascade = cv2.CascadeClassifier("data/haarcascade_eye_tree_eyeglasses.xml")

image, gray = preprocessing()

if image is None: raise Exception("영상 파일 읽기 에러")

faces = face_cascade.detectMultiScale(gray, 1.1, 2, 0, (100, 100))

if faces.any():

    # 얼굴 위치 값을 가져오기
    x, y, w, h = faces[0]

    # 원본이미지로부터 얼굴영역 가져오기
    face_image = image[y:y + h, x:x + w]

    eyes = eye_cascade.detectMultiScale(face_image, 1.15, 7, 0, (25, 20))

    if len(eyes) == 2:
        for ex, ey, ew, eh in eyes:
            center = (x + ex + ew // 2, y + ey + eh // 2)

            cv2.circle(image, center, 10, (0, 255, 0), 2)

    else:
        print("눈 미검출")

    cv2.rectangle(image, faces[0], (255, 0, 0), 4)

    cv2.imshow("MyFace", image)

else: print("얼굴 미검출")

cv2.waitKey(0)

