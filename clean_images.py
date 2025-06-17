import os  # 폴더와 파일을 다루는 도구다
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # info/warning 메시지 숨긴다
import warnings  # 경고 메시지를 다루는 도구다
warnings.filterwarnings('ignore')  # 경고 메시지를 안 보이게 한다
from PIL import Image  # 이미지를 다루는 도구다

# 손상된 이미지를 찾아서 삭제하는 함수다
def clean_broken_images(folder):
    for subdir in os.listdir(folder):  # 폴더 안에 있는 모든 폴더를 돈다
        subfolder = os.path.join(folder, subdir)
        if not os.path.isdir(subfolder):  # 폴더가 아니면 넘어간다
            continue
        for fname in os.listdir(subfolder):  # 폴더 안에 있는 모든 파일을 돈다
            fpath = os.path.join(subfolder, fname)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    with Image.open(fpath) as img:
                        img.verify()  # 이미지가 제대로 된 파일인지 확인한다
            except Exception:
                print('삭제:', fpath)  # 문제가 있으면 파일을 삭제한다
                os.remove(fpath)

if __name__ == "__main__":
    train_dir = r'E:\Pet Image Classification\PetImages'  # 훈련용 사진 폴더다
    validation_dir = r'E:\Pet Image Classification\PetImagesTest'  # 테스트용 사진 폴더다
    clean_broken_images(train_dir)  # 훈련용 폴더에서 손상된 이미지를 지운다
    clean_broken_images(validation_dir)  # 테스트용 폴더에서도 손상된 이미지를 지운다
    print('손상된 이미지 정리 완료다')