import os
import random
import cv2
import numpy as np

train_path = 'C:/Users/remo/Desktop/datas/train/'
label_path = 'C:/Users/remo/Desktop/datas/train/MASK/'
file_list = os.listdir(train_path)

#.split -> .jpg제거 / .png붙여서 해결.

# 1. train data가 저장된 곳에서 train_path를 txt에 쓰기.

file = open(train_path + "train.txt", 'w')
for idx in range(len(file_list)-2):
    file.write(train_path + file_list[idx] + '\n')
file.close()

# 1-1. label data가 저장된 곳에서 mask_path를 txt에 쓰기.
file = open(label_path + "mask.txt", 'w')
for idx in range(len(file_list)-2):
    file.write(label_path + file_list[idx].replace('jpg', 'png') + '\n')    # .jpg -> .png
file.close()

# 2. 각 .txt를 읽어와서, idx를 맞추고, augmentation하기.
def load_images():
    augmented_imgs = []
    augmented_lbls = []
    train_path_list = []
    label_path_list = []

    for idx in range(len(file_list)-2):
        file = open(train_path + "train.txt")
        filename = file.readlines()
        filename = filename[idx].rstrip()   # .rstrip() -> 마지막 개행을 삭제해준다.
        train_path_list.append(filename)

    for idx in range(len(file_list) - 2):
        file = open(label_path + "mask.txt")
        filename = file.readlines()
        filename = filename[idx].rstrip()   # .rstrip() -> 마지막 개행을 삭제해준다.
        label_path_list.append(filename)

    for idx in range(0, len(train_path_list)-1):
        img_path = train_path_list
        lbl_path = label_path_list
        img = cv2.imread(str(img_path[idx]), cv2.IMREAD_COLOR)
        # __여기서 train image crop__
        img = img[0:720, 168:1105]
        img = cv2.resize(img, (256, 256))

        lbl = cv2.imread(str(lbl_path[idx]), cv2.IMREAD_COLOR)
        # __여기서 label image crop__
        lbl = lbl[0:720, 168:1105]
        lbl = cv2.resize(lbl, (256, 256))

        rand = random.random()
        if rand < 0.5:
            img = aug_flip_bright(img)
            lbl = aug_flip(lbl)
            img = (img / 255.).astype(np.float32)
            #lbl = (lbl / 255.).astype(np.float32)
        else:
            img = img
            lbl = lbl
            img = (img / 255.).astype(np.float32)
            #lbl = (lbl / 255.).astype(np.float32)
        '''
        cv2.imshow('img', img)
        cv2.imshow('lbl', lbl)
        cv2.waitKey(0)'''

        augmented_imgs.append(img)
        augmented_lbls.append(lbl)

    return augmented_imgs, augmented_lbls

# 3. Image Augementation
def aug_flip_bright(img):

    img = cv2.transpose(img)
    img = cv2.flip(img, 1)
    img = cv2.add(img, 50)

    return img

def aug_flip(lbl):

    lbl = cv2.transpose(lbl)
    lbl = cv2.flip(lbl, 1)

    return lbl

# 4. Random idx로 train[idx], label[idx] 매칭하기.
def random_match(augmented_imgs, augmented_lbls):
    zipped_list = []
    len_idx = 392

    for idx in range(0, len_idx):
        rand = random.randint(0, 312)
        zipped_list.append([augmented_imgs[rand], augmented_lbls[rand]])

    return zipped_list, len_idx

'''
# crop 결과물
for idx in range(0, len(augmented_imgs)):
    #print(idx)
    cv2.imshow('train', augmented_imgs[idx])
    cv2.imshow('mask', augmented_lbls[idx])
    cv2.waitKey(0)
'''

if __name__ == "__main__":
    load_images()
    print("hello from data handler")