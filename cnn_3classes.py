import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# 1. CIFAR-10 데이터셋 다운로드
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 2. 사용할 클래스 인덱스 (cat=3, dog=5, frog=6)
target_classes = [3, 5, 6]

# 3. 해당 클래스만 추출
def filter_classes(x, y, targets):
    idx = np.isin(y, targets).flatten()
    x = x[idx]
    y = y[idx]
    y = np.array([targets.index(label) for label in y.flatten()])  # 0,1,2로 변환
    return x, y

x_train_f, y_train_f = filter_classes(x_train, y_train, target_classes)
x_test_f, y_test_f = filter_classes(x_test, y_test, target_classes)

# 4. 정규화 및 원-핫 인코딩
x_train_f = x_train_f.astype('float32') / 255.
x_test_f = x_test_f.astype('float32') / 255.
y_train_f = to_categorical(y_train_f, num_classes=3)
y_test_f = to_categorical(y_test_f, num_classes=3)

# 5. 모델 정의
model = models.Sequential()
model.add(layers.Input(shape=(32, 32, 3)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))  # 3개 클래스

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 6. 모델 학습
history = model.fit(
    x_train_f, y_train_f,
    epochs=10,
    batch_size=32,
    validation_data=(x_test_f, y_test_f)
)

# 7. 모델 저장
model.save('cifar10_3class_model.keras')
print('모델이 cifar10_3class_model.keras 파일로 저장됨')
