import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # info/warning 메시지 숨김
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
import os
import warnings

def clean_broken_images(folder):
    for subdir in os.listdir(folder):
        subfolder = os.path.join(folder, subdir)
        if not os.path.isdir(subfolder):
            continue
        for fname in os.listdir(subfolder):
            fpath = os.path.join(subfolder, fname)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    with Image.open(fpath) as img:
                        img.verify()
            except Exception:
                print('삭제:', fpath)
                os.remove(fpath)

if __name__ == "__main__":
    train_dir = r'E:\Pet Image Classification\PetImages'
    validation_dir = r'E:\Pet Image Classification\PetImagesTest'
    clean_broken_images(train_dir)
    clean_broken_images(validation_dir)
    print('손상된 이미지 정리 완료')