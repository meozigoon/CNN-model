import os
import numpy as np
import keras
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras import layers, models
from keras.utils import load_img, img_to_array
import itertools

train_dir = r'E:\\Pet Image Classification\\PetImages'
validation_dir = r'E:\\Pet Image Classification\\PetImagesTest'

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

def repeat_generator(generator):
    while True:
        for batch in generator:
            yield batch

train_gen_repeat = repeat_generator(train_generator)
val_gen_repeat = repeat_generator(validation_generator)

model = models.Sequential()
model.add(layers.Input(shape=(150, 150, 3)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

model.save('my_model.h5')
print('모델이 my_model.h5 파일로 저장되었습니다.')