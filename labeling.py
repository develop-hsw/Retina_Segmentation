from code import data_handler as dh
import cv2
import numpy as np

#np.set_printoptions(threshold=np.inf, linewidth=np.inf) # -> 출력 생략(...) 안하는 설정
from copy import deepcopy
augmented_imgs, augmented_lbls = dh.load_images()
zipped_list, len_idx = dh.random_match(augmented_imgs, augmented_lbls) # -> (aug_img, aug_lbl) 형태

#print(np.shape(augmented_imgs(lbls))) # -> (392, 512, 512, 3)

# 임계값
def make_mask():
    mask_list = []

    # 1. 색상별 범위
    lower_blue = np.array([200, 0, 0])
    upper_blue = np.array([255, 50, 50])

    lower_green = np.array([0, 200, 0])
    upper_green = np.array([50, 255, 50])

    lower_red = np.array([0, 0, 200])
    upper_red = np.array([50, 50, 255])

    lower_white = np.array([200, 200, 200])
    upper_white = np.array([255, 255, 255])

    # 2. mask 만들기.
    for idx in range(0, 392):
        mask_blue = cv2.inRange(zipped_list[idx][1], lower_blue, upper_blue)/255
        mask_green = cv2.inRange(zipped_list[idx][1], lower_green, upper_green)/255
        mask_red = cv2.inRange(zipped_list[idx][1], lower_red, upper_red)/255
        mask_white = cv2.inRange(zipped_list[idx][1], lower_white, upper_white) / 255
        """
        # label_blue
        label_blue = np.zeros([512, 512])
        label_blue += mask_blue
        label_blue.astype(np.uint8)

        # label_green
        label_green = np.zeros([512, 512])
        label_green += mask_green
        label_green.astype(np.uint8)

        # label_red
        label_red = np.zeros([512, 512])
        label_red += mask_red
        label_red.astype(np.uint8)

        # label - bgr = backgrond(label_white)
        label_sum = np.zeros([512, 512])
        label_sum += label_blue
        label_sum += label_green
        label_sum += label_red
        #print(np.max(label_sum),np.min(label_sum))
        label_white = np.ones([512, 512])
        label_white -= label_sum
        label_white.astype(np.uint8)
        #print(np.max(label_white),np.min(label_white))
        """
        mask_pad = np.zeros([256, 256, 4])
        mask_pad[:,:,0] = mask_white
        mask_pad[:,:,1] = mask_blue
        mask_pad[:,:,2] = mask_green
        mask_pad[:,:,3] = mask_red
        #mask_pad /= 255
        mask_list.append(mask_pad)

        # 이 부분 문법 파악하기.
        # argmax에서 axis가 다뤄지는 부분.
        # 테스트 코드일 뿐.
        mask_arg = deepcopy(np.argmax(mask_pad, axis=2))
        mask_arg = mask_arg * 50
        mask_arg = np.array(mask_arg,dtype=np.uint8)

        '''
        cv2.imshow('blue', label_blue)
        cv2.imshow('green', label_green)
        cv2.imshow('red', label_red)
        cv2.imshow('origin', zipped_list[idx][0])
        cv2.imshow('label', zipped_list[idx][1])
        cv2.imshow('test', mask_arg)
        cv2.waitKey(0)
        '''

    return zipped_list, mask_list


if __name__ == "__main__":
    print("*** labeling ***")