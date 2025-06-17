from keras.models import load_model  # 저장된 모델을 불러오는 도구다
from keras.utils import load_img, img_to_array  # 이미지를 불러오고 배열로 바꿔주는 도구다
import numpy as np  # 숫자 계산을 쉽게 해주는 도구다

def predict_image(img_path, model_path='my_model.keras'):
    model = load_model(model_path)  # 저장된 모델을 불러온다
    img = load_img(img_path, target_size=(150, 150))  # 이미지를 150x150 크기로 불러온다
    img_array = img_to_array(img) / 255.  # 이미지를 숫자 배열로 바꾸고 0~1로 만든다
    img_array = np.expand_dims(img_array, axis=0)  # 배열 모양을 바꿔서 모델이 예측할 수 있게 한다
    prediction = model.predict(img_array)  # 모델로 이미지를 예측한다
    if prediction[0][0] > 0.5:
        print(f"{img_path}: 개다")
    else:
        print(f"{img_path}: 고양이다")

if __name__ == "__main__":
    # 아래 경로를 원하는 이미지로 바꿔서 사용한다
    predict_image(r'E:\Pet Image Classification\PetImagesTest\Dog\dogTest0.jpg')
