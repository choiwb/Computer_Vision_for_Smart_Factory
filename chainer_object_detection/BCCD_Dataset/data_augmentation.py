from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

import matplotlib.pyplot as plt
import os
import time
import glob

# original image data folder
original_data_path = 'C:/Users/md459/PycharmProjects/choiwb/BigData_Team_AI_Contest/BCCD_Dataset/BCCD/JPEGImages'

# saving data augmentation image folder
save_aug = 'C:/Users/md459/PycharmProjects/choiwb/BigData_Team_AI_Contest/BCCD_Dataset/BCCD/save_aug'

'''
# change jpg file name
os.chdir(original_data_path)
for index, oldfile in enumerate(glob.glob("*.jpg"), start=1):
    # newfile = '{}.jpg'.format(index)
    newfile = 'coffee_{}.jpg'.format(index + 1)
    newfile = 'coffee_{:03}.jpg'.format(index)
    os.rename (oldfile,newfile)
'''

datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

fnames = sorted([os.path.join(original_data_path, fname) for fname in os.listdir(original_data_path)])
print(fnames)

start = time.time()

# 이미지 전처리 유틸리티 모듈
# original image data folder: 총 i 개 이미지
for i in range(len(fnames)):

    # 증식할 이미지 선택합니다
    img_path = fnames[i]

    # 이미지를 읽고 크기를 변경합니다
    # img = image.load_img(img_path, target_size=(150, 150))
    img = image.load_img(img_path)

    # (150, 150, 3) 크기의 넘파이 배열로 변환합니다
    x = image.img_to_array(img)

    # (1, 150, 150, 3) 크기로 변환합니다
    x = x.reshape((1,) + x.shape)

    # flow() 메서드는 랜덤하게 변환된 이미지의 배치를 생성합니다.
    # 무한 반복되기 때문에 어느 지점에서 중지해야 합니다!
    j = 1
    for batch in datagen.flow(x, batch_size=1):
        plt.figure(i)
        imgplot = plt.imshow(image.array_to_img(batch[0]))
        plt.savefig(os.path.join(save_aug, 'aug_%d_%d.jpg' %(i+1, j)))

        j += 1
        # 한 이미지당 4번 증식 후, 종료
        if j % 5 == 0:
            break

    # plt.show()

end = time.time()
operation = end - start

print('4배 증식 걸린 시간 : %.3f (초)' %(operation))