# 숫자 계산을 쉽게 해주는 도구를 가져온다
import os  # 폴더와 파일을 다루는 도구를 가져온다
import numpy as np  # 숫자 배열을 쉽게 다루는 도구를 가져온다
import keras  # 딥러닝을 쉽게 해주는 도구를 가져온다
from keras.src.legacy.preprocessing.image import ImageDataGenerator  # 이미지를 쉽게 불러오고 변형하는 도구를 가져온다
from keras import layers, models  # 신경망의 층과 모델을 만드는 도구를 가져온다
from keras.utils import load_img, img_to_array  # 이미지를 불러오고 숫자 배열로 바꿔주는 도구를 가져온다

# 학습에 사용할 사진들이 들어있는 폴더 경로를 쓴다
train_dir = r'E:\Pet Image Classification\PetImages'  # 훈련용 사진 폴더
validation_dir = r'E:\Pet Image Classification\PetImagesTest'  # 테스트용 사진 폴더

# 이미지를 0~1 사이 숫자로 바꿔주는 도구를 만든다
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

# 훈련용 사진을 불러오고, 크기를 150x150으로 맞춘다
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # 사진 크기를 150x150으로 바꾼다
    batch_size=32,  # 한 번에 32장씩 가져온다
    class_mode='binary'  # 고양이/개 두 가지로 분류한다
)

# 테스트용 사진도 똑같이 불러온다
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# 신경망 모델을 만든다
model = models.Sequential()  # 층을 차곡차곡 쌓는 방식이다
model.add(layers.Input(shape=(150, 150, 3)))  # 150x150 크기의 컬러 사진을 입력으로 받는다
model.add(layers.Conv2D(32, (3, 3), activation='relu'))  # 사진에서 특징을 찾는 층이다
model.add(layers.MaxPooling2D((2, 2)))  # 중요한 정보만 남기고 줄여준다
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())  # 2차원 이미지를 1차원으로 펴준다
model.add(layers.Dense(512, activation='relu'))  # 똑똑한 계산을 해주는 층이다
model.add(layers.Dense(1, activation='sigmoid'))  # 결과를 0~1 사이로 만들어준다(고양이/개)

# 모델을 학습할 준비를 한다
model.compile(
    loss='binary_crossentropy',  # 정답과 예측이 얼마나 다른지 계산하는 방법이다
    optimizer='adam',  # 똑똑하게 학습하는 방법이다
    metrics=['accuracy']  # 얼마나 잘 맞추는지 확인한다
)

# 모델을 학습시킨다
history = model.fit(
    train_generator,  # 훈련용 사진을 사용한다
    epochs=10,  # 10번 반복해서 학습한다
    validation_data=validation_generator  # 테스트용 사진으로도 확인한다
)

# 학습이 끝난 모델을 파일로 저장한다
model.save('my_model.keras')
print('모델이 my_model.keras 파일로 저장되었다')