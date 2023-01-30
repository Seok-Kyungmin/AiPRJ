import pandas as pd
import numpy as np
import pickle

from keras.saving.legacy.save import load_model
from konlpy.tag import Okt
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Dense, LSTM
from keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint



#학습용 데이터
train_data = pd.read_table("data/ratings_train.txt")

# 생성할 학습 모델의 평가를 테스트할 데이터
test_data = pd.read_table("data/ratings_test.txt")

print("학습용 데이터의 네이버영화 리뷰 개수 : ", len(train_data))
print("테스트 데이터의 네이버영화 리뷰 개수 : ", len(test_data))

# 데이터가 정상적으로 가져오는지 확인하기 위해 상위 10갸 출력
print(train_data[:10])
print(test_data[:10])

# 학습용 데이터 정체 시작

#document의 리뷰 내용과  lebel의 긍정, 부정 레코드의 중복이 존재하는지 확인
print("중복 제거된 학습용 데이터 수 확인 : ", train_data["document"].nuniqui(), train_data["label"].nunique())

train_data.drop_duplicates(subset=["document"], inplace=True)

print("중복 제거된 최종 학습용 데이터 수 : ", len(train_data))

print(train_data.groupby("label").size().reset_index(name = "count"))

print("널 값이 존재하는 학습용 데이터 확인 : ", train_data.isnull().values.any())

print("널값이 존재하는 학습용 데이터 수")
print(train_data.isnull().sum())

print("널값인 데이터 확인")
print(train_data.loc[train_data.document.isnull()])

train_data = train_data.dropna(how="any")
print("널 값이 존재하는 학습용 데이터 다시 확인 : ", train_data.isnull().values.any())

print("널값이 제거된 최종 학습용 데이터 수 : ",len(train_data))

train_data["document"] = train_data["document"].str.replace("[^ㄱ-하-ㅣ가-힝","")

train_data["document"] = train_data["document"].str.replace('^ +', "")

print(train_data["document"].replace("",np.nan, inplace=True))

print("널값이 존재하는 학습용 데이터 다시 확인")
print(train_data.isnull().sum())

train_data = train_data.dropna(how='any')
print("한글 외 단어 및 널값이 제거된 최종 학습용 데이터 수 : ", len(train_data))

# 테스트용 데이터 정체 시작

print("중복 제거된 학습용 데이터 수 확인 : ", test_data["document"].nuniqui(), test_data["label"].nunique())

train_data.drop_duplicates(subset=["document"], inplace=True)

print("중복 제거된 최종 학습용 데이터 수 : ", len(test_data))

print(train_data.groupby("label").size().reset_index(name = "count"))

print("널 값이 존재하는 학습용 데이터 확인 : ", test_data.isnull().values.any())

print("널값이 존재하는 학습용 데이터 수")
print(test_data.isnull().sum())

print("널값인 데이터 확인")
print(test_data.loc[test_data.document.isnull()])

test_data = test_data.dropna(how="any")
print("널 값이 존재하는 학습용 데이터 다시 확인 : ", test_data.isnull().values.any())

print("널값이 제거된 최종 학습용 데이터 수 : ", len(test_data))

test_data["document"] = test_data["document"].str.replace("[^ㄱ-하-ㅣ가-힝","")

test_data["document"] = test_data["document"].str.replace('^ +', "")

print(test_data["document"].replace("", np.nan, inplace=True))

print("널값이 존재하는 학습용 데이터 다시 확인")
print(test_data.isnull().sum())

test_data = test_data.dropna(how='any')
print("한글 외 단어 및 널값이 제거된 최종 학습용 데이터 수 : ", len(test_data))

# 임시로 만든 불용어 사전
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

okt = Okt()

X_train = []
for sentence in tqdm(train_data["document"]):
    tokenized_sentence = okt.morphs(sentence, stem=True)
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]
    X_train.append(stopwords_removed_sentence)

X_test = []
for sentence in tqdm(test_data['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True)
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]
    X_test.append(stopwords_removed_sentence)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

threshold = 3

total_cnt = len(tokenizer.word_index)

rare_cnt = 0

total_freq = 0

rare_freq = 0

for key, value in tokenizer.word_counts.items():
    total_freq = total_freq +value

    if(value < threshold):
        rare_cnt = rare_cnt +1
        rare_freq = rare_freq + value

print('단어 집학(vocabulary)의 크기 : ', total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold -1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

vocab_size = total_cnt - rare_cnt + 1
print('단어 집합의 크기 :',vocab_size)

tokenizer = Tokenizer(vocab_size)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

with open("model/naver_movie_tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle)

y_train = np.array(train_data["label"])
y_test = np.array(test_data["label"])

drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]

X_train = np.delete(X_train, drop_train, axis=0)
y_train = np.delete(y_train, drop_train, axis=0)

def below_threshold_len(max_len, nested_list):
    count = 0
    for sentence in nested_list:
        if(len(sentence) <= max_len):
            count = count +1
    print("전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s" %(max_len, (count / len(nested_list))*100))

max_len = 30
below_threshold_len(max_len, X_train)

X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)

embedding_dim = 100
hidden_units = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(LSTM(hidden_units))
model.add(Dense(1, activation="sigmoid"))

es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=4)
mc = ModelCheckpoint("naver_movie.h5", monitor="val_acc", mode="max", verbose=1, save_best_only=True)

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=64, validation_split=0.2)

loaded_model = load_model("model/naver_movie.h5")
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test)[1]))

