text = "연합뉴스 한국폴리텍대학 서울강서캠퍼스(학장 노정진)는 데이터분석과 2학년 김한결, 황윤영, 홍두표 학생과 1학년 김도우 학생이 " \
       "한이음 공모전에서 장려상을 받아 세종대학교에서 개최된 '소프트웨어(SW) 인재페스티벌'에 초청받았다고 10일 밝혔다." \
       "과학기술정보통신부가 주최한 SW 인재페스티벌은 한이음 공모전 수상작에 대한 시상식이 진행됐다." \
       "이번에 수상한 작품은 빅데이터 분석을 활용한 음성 인식 스피치 교정 애플리케이션으로 최근 블라인드 채용으로 인해" \
       "면접에 대한 중요성이 강조되면서, 면접 교육을 위한 수강료 등 다양한 비용을 감소시키기 위해 개발한 소프트웨어다." \
       "사용자는 태블릿, 스마트폰과 같은 스마트 기기로부터 면접 질문을 받아 마이크에 음성으로 면접 질문에 대해 대답하고, " \
       "소프트웨어는 그 대답에 대한 분석을 진행한다. 분석 내용은 자주 사용하는 명사 단어, 목소리의 높낮이, 목소리 빠르기, " \
       "긍정·부정 단어 사용 여부 등이며, 분석을 위해 한글 형태소 분석, 음성인식(TTS), 오피니언 마이닝 등 기술이 적용됐다."

# 예시 문장 출력
print(text)

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

#stopwords 변수에 원하지 않는 단어들 추가
stopwords = set(STOPWORDS)
stopwords.add("위해")
stopwords.add("받아")
stopwords.add("분석을")

myWC = WordCloud(font_path="font/NanumGothicCoding.ttf", stopwords=stopwords, background_color="white").generate(text)

plt.figure(figsize=(5, 5))

plt.imshow(myWC, interpolation="lanczos")

plt.axis('off')

plt.show()
