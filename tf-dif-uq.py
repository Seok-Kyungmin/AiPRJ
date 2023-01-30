import operator

from math import log

import numpy as np
from numpy import dot
from numpy.linalg import norm

from konlpy.tag import Hannanum

import re

def tf(t, d):
    return d.count(t)

def idf(t):
    df = 0
    for doc in docs:
        df += t in doc

    return log(N/(df + 1))

def tfidf(t, d):
    return tf(t, d) * idf(t)

def dist(x, y):
    return np.sqrt(np.sum((x-y)**2))

myHannanum = Hannanum()

org_docs = [
    "학생들은 빅데이터와 인공지능 기술을 배우고 있다.",
    "빅데이터 기술은 방대한 데이터를 처리한다. 빅데이터는 많은 데이터를 저장한다.",
    "빅데이터 기술은 많이 어럽다. 특히 하둡이 어렵다.",
    "나의 목표는 빅데이터 기술을 활용하는 빅데이터 소프트웨어 개발자이다.",
    "소프트웨어 개발은 코딩이 필수이다. 나는 소프트웨어 개발자가 되고 싶다. 소프트웨어 개발자 화이팅!",
    "인공지능 기술에서 자연어 처리는 재미있다. 자연어는 사람이 사용하는 일반적인 언어이다."

]

docs = []

for org_doc in org_docs:
    replace_doc = re.sub("[!@#$%^&*()_+]", " ", org_doc)
    docs.append(" ".join(myHannanum.nouns(replace_doc)))

print(docs)

vocab = list(set(w for doc in docs for w in doc.split()))
vocab.sort()

print("중복 제거된 단어 : " + str(vocab))

N = len(docs)

print("문서의 수 : " + str(N))

result = []
for i in range(N):
    result.append([])
    d = docs[i]

    for j in range(len(vocab)):
        t = vocab[j]
        result[-1].append(tfidf(t, d))

    print(result[i])

print("문서1과 문서2의  유사도 : " + str(dist(np.array(result[0]), np.array(result[1]))))

print("문서1과 문서3의  유사도 : " + str(dist(np.array(result[0]), np.array(result[2]))))

print("문서1과 문서4의  유사도 : " + str(dist(np.array(result[0]), np.array(result[3]))))

print("문서1과 문서5의  유사도 : " + str(dist(np.array(result[0]), np.array(result[4]))))

print("문서1과 문서6의  유사도 : " + str(dist(np.array(result[0]), np.array(result[5]))))

print("-----------------------------")

#유사도 분석 결과를 저장하기 위해 dic객체 선언
res = {}

#문서1과 유사한 문서는?
for i in range(N):

    doc_number = i+1

    if doc_number!=1:

        u_res = dist(np.array(result[0]), np.array(result[i]))

        res[i] = u_res

        print("문서1과 문서" + str(doc_number) + "의 유사도 : " + str(u_res))

print("-----------------------------")
print("결과값 : " + str(res))
print("-----------------------------")
print("문서1과 가장 유사한 문서는? : ")

my_doc = sorted(res.items(), key=operator.itemgetter(1), reverse=True)

print("결과 : " + str(my_doc[0]))
print("문서1 : " + str(org_docs[0]))
print("문서" + str(my_doc[0][0] + 1) + " : " + str(org_docs[my_doc[0][0]]))