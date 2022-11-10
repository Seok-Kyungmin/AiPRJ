from konlpy.tag import Okt
import re
okt=Okt()

# .제거
token=re.sub("(\.)","","한국폴리텍대학 서울강서캠퍼스 데이터분석과 이협건 교수는 "
             "한국폴리텍대학에서 데이터분석 과목과 인공지능 과목을 교육하는 교수이다.")

# 형태소 단위로 나누기
token=okt.morphs(token)

word2index={} # 인덱스 저장
bow=[] #bow 저장

for word in token:

    # word2index에 값이 존재하지 않는 경우
    if word not in word2index.keys():

        # 값 넣기
        word2index[word]=len(word2index)

        # Bow 전체에 전부 기본값 1을 넣어줍니다. 단언의 개수는 최소 1개 이상이기 때문이다
        bow.insert(len(word2index)-1,1)

    else:
        # 재등장하는 단어의 인덱스 가져오기
        index=word2index.get(word)

        bow[index]=bow[index]+1

print(word2index)

print(bow)