from keras.models import load_model  # 저장된 모델을 불러오는 도구다
from keras.utils import load_img, img_to_array  # 이미지를 불러오고 배열로 바꿔주는 도구다
import numpy as np  # 숫자 계산을 쉽게 해주는 도구다

# 3개 클래스명
class_names = ['고양이', '개', '개구리']  # [cat, dog, frog]

def predict_image(img_path, model_path='cifar10_3class_model.keras'):
    model = load_model(model_path)  # 저장된 모델을 불러온다
    img = load_img(img_path, target_size=(32, 32))  # CIFAR-10 이미지 크기에 맞게 32x32로 불러온다
    img_array = img_to_array(img) / 255.  # 이미지를 숫자 배열로 바꾸고 0~1로 만든다
    img_array = np.expand_dims(img_array, axis=0)  # (1, 32, 32, 3)
    prediction = model.predict(img_array)  # shape: (1, 3)
    pred_class = np.argmax(prediction[0])  # 확률이 가장 큰 클래스
    print(f"{img_path}: {class_names[pred_class]} (확률: {prediction[0][pred_class]:.2f})")

if __name__ == "__main__":
    # 아래 경로를 원하는 이미지로 바꿔서 사용한다
    predict_image(r'E:\Pet Image Classification\PetImagesTest\Cat\cat_124.jpg')
