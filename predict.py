from keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np

def predict_image(img_path, model_path='my_model.h5'):
    model = load_model(model_path)
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        print(f"{img_path}: 개입니다.")
    else:
        print(f"{img_path}: 고양이입니다.")

if __name__ == "__main__":
    predict_image(r'E:\Pet Image Classification\PetImagesTest\Cat\catTest0.jpg')
