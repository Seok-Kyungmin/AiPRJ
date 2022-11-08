# 워드 클라우드의 형태를 꾸미기 위해 기능 추가
import numpy as np

# 이미지 제이를 위해 PIL 라이브러리로부터 Image 기능 사용
from PIL import Image

# 설치한 wordcloud 외부라이브러리로부터 WordCloud 기능 사용하도록 설정
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

from konlpy.tag import Hannanum

import re

text = open("contents.txt", encoding="UTF-8").read()

print(text)

myHannanum = Hannanum()

# 단어 분석의 정확도를 높이기 위해 특수문자 제거
# 특수 문자는 키보드 상단 숫자패드의 특수문자가 발견되면 한칸 공백으로 변경
replace_text = re.sub("[!@#$%^&*()_+]", " ", text)

# 특수문자가 제거된 문장 출력
print(replace_text)

# 명사 분석 결과는 여러 단어들이 저장된 배열형태로 데이터를 생성하기 때문에 배열을 문자열로 변경하기 위해
# join 함수를 사용하며, analysis_text 변수에 문자열로 변횐된 결과를 저장함
analysis_text = (" ".join(myHannanum.nouns(replace_text)))

#stopwords 변수에 원하지 않는 단어들 추가
stopwords = set(STOPWORDS)
# stopwords.add("분석")
# stopwords.add("소프트웨어")

# 워드 클라우드 형태 이미지 가져오기
myImg = np.array(Image.open("image/lalf.jpg"))

imgColor = ImageColorGenerator(myImg)

myWC = WordCloud(font_path="font/NanumGothicCoding.ttf", mask=myImg, stopwords=stopwords, background_color="white").generate(analysis_text)

plt.figure(figsize=(5, 5))

plt.imshow(myWC.recolor(color_func=imgColor), interpolation="lanczos")

plt.axis('off')

plt.show()