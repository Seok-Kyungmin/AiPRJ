from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib

data_dir = pathlib.Path("images")
print("data_dir", data_dir)

image_count = len(list(data_dir.glob("*/*.jpg")))
print("image_count : ", image_count)

# 1. 데이터 세트 만들기

#로더에 대한 몇가지 매개 변수를 정의
batch_size = 128
img_height = 180
img_width = 100

# 2. 학습 모델 만들기
# 일반적으로 학습 모델을 만들때 학습데이터의 80%는 학습용, 20%는 점증용으로 사용함

#학습용 데이터셋 생성(검증용 20% 사용)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    send = 123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    send=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# 각 꽃들의 폴더 이름을 객체 이름으로 사용
class_names = train_ds.class_names
print("class_names", class_names)

for image_batch, labels_batch in train_ds:
    print("image_batch", image_batch.shape)
    print("labels_batch", labels_batch.shape)
    break

normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch,labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

num_classes = 5

# 3. 학습 모델 구성
# 학습을 위한 신경망 구성
model = Sequential([
    # 이미지 전처리
    layers.experimental.preprocessing.Rescaling(1./255, input_shape= (img_height,img_width, 3)),

    layers.Conv2D(16, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),

    layers.Dense(num_classes)
])

# 4. 학습 모델 컴파일
# 일반적으로 학습 모델을 만들때 학습데이터의 80%는 학습용, 20%는 검증용으로 사용함
model.compile(optimizer="adam",
              loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

model.summary()

epochs=10

# 5. 학습 수행
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# 6. 학습모델 파일 생성
model.save("medel/myFlower1.h5")

# 7. 학습 결과 시각화
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss =history.history["val_loss"]

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.show()


