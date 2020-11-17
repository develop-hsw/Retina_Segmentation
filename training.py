from code import labeling as lbl
import tensorflow as tf
import numpy as np
from code.network import Network
import cv2
import os
from copy import deepcopy
#np.set_printoptions(threshold=np.inf, linewidth=np.inf)

zipped_list, mask_list = lbl.make_mask()
print("mask_list", np.shape(mask_list))
save_path = 'C:/Users/remo/Desktop/img_save/'

train_set = []
train_mask = []

valid_set = []
valid_mask = []


# make train / validation
for idx in range(0, 392): # 80%
    if(idx <= 311):
        tmp_img = np.array(zipped_list[idx][0])/127.5-1
        train_set.append(tmp_img)
        train_mask.append(mask_list[idx])
        #print(train_mask[idx])

    else:
        tmp_img = np.array(zipped_list[idx][0]) / 127.5 - 1
        valid_set.append(tmp_img)
        valid_mask.append(mask_list[idx])

# class<list> => class<numpy.array>
train_set = np.array(train_set)
train_mask = np.array(train_mask)

valid_set = np.array(valid_set)
valid_mask = np.array(valid_mask)

OUTPUT_CHANNELS = 4 # default / R / G / B -> 4개

# len(train_set) 312 / iteration 39 = batch_size 8
batch_size = 8
iteration = len(train_set) / batch_size
iteration = int(iteration)

image_size = 512
lr = 0.001 # learning rate
gpu_fraction = 0.9 # gpu 사용률. 100%는 안되고, 최대가 90%.


#train_set = np.resize(train_set, [batch_size, 512, 512, 3])
#train_mask = np.resize(train_mask, [batch_size, 512, 512, 4])

def train():
    # input_photo 와 label은 형식이 지정된 텐서(placeholder = 틀)이다.
    input_photo = tf.placeholder(tf.float32, [batch_size, image_size, image_size, 3])
    label = tf.placeholder(tf.float32, [batch_size, image_size, image_size, 4])
    output = Network(input_photo, class_num=OUTPUT_CHANNELS).model

    #recon_loss = tf.reduce_mean(tf.losses.absolute_difference(label, output))
    cost = tf.reduce_mean(tf.losses.absolute_difference(label, output))
    correct_prediction = tf.equal(tf.argmax(output, -1), tf.argmax(label, -1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # loss = tf.sqrt(tf.reduce_mean(tf.square(output - label)))

    optim = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.99).minimize(cost)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    with tf.device('gpu:0'): # 하단 설정 내용.
        sess.run(tf.global_variables_initializer())

        for epoch in range(0, 100):  # epoch을 얼마나 돌릴지 임의로 저장.
            for idx in range(0, int(len(train_set)/batch_size)):
                # iteration = int(len(train_set)/batch_size)
                # 내가 생각한 것은 -> epoch 0 : idx / epoch >=1 : epoch * batch_size + idx
                # 근데 그냥 슬라이싱을 쓰면... 된다...

                photo_batch = deepcopy(train_set[idx * batch_size : (idx+1) * batch_size])
                photo_batch = np.resize(photo_batch, [8, 512, 512, 3])
                photo_batch = np.array(photo_batch)
                label_batch = deepcopy(train_mask[idx * batch_size : (idx+1) * batch_size])
                label_batch = np.resize(label_batch, [8, 512, 512, 4])
                label_batch = np.array(label_batch)

                #print(np.shape(label_batch),type(label_batch))
                """
                for i in range(0,8):
                    img = photo_batch[i]
                    lbl = label_batch[i]

                    for j in range(0,4):
                        tmp_lbl = lbl[:,:,j]
                        tmp_lbl *=100
                        tmp_lbl = np.array(tmp_lbl,dtype=np.uint8)
                        cv2.imshow(str(j),tmp_lbl)
                    lbl = np.array(np.argmax(lbl,-1)) *50
                    lbl = lbl.astype(np.uint8)
                    cv2.imshow("img",img)
                    cv2.imshow("lbl",lbl)
                    cv2.waitKey()
                """
                _, r_loss = sess.run([optim, cost], feed_dict={input_photo: photo_batch, label: label_batch})

            print("***", epoch, "***")

            #for idx in range(0, int(len(valid_set)/batch_size)):
            for idx in range(0, int(len(valid_set)/batch_size)):
                photo_valid = valid_set[idx * batch_size : (idx+1) * batch_size]
                photo_valid = np.resize(photo_valid, [8, 512, 512, 3] )

                '''
                input_valid = photo_valid
                input_valid = np.resize(input_valid, [8, 512, 512, 3])
                input_valid = np.array(input_valid, dtype=np.uint8)
                '''
                label_valid = valid_mask[idx * batch_size : (idx+1) * batch_size]
                label_valid = np.resize(label_valid, [8, 512, 512, 4])


                result_valid, acc_valid = sess.run([output, accuracy], feed_dict={input_photo: photo_valid, label: label_valid})
                #print(np.max(result_valid))

                cv2.waitKey()

                result = np.argmax(result_valid, -1)
                result = np.array(result, dtype=np.uint8)
                result = result *50


                label_valid = np.resize(label_valid, [8, 512, 512])
                label_valid = np.array(label_valid, dtype=np.uint8)
                #print("lv.np", np.shape(label_valid))

                #intersection, iou_result = iou.iou_evlt(label_valid,result)
                for i in range(0,batch_size):
                    tmp_result = result_valid[i]
                    tmp_result = np.argmax(tmp_result, -1)
                    tmp_result = tmp_result*50
                    tmp_result = np.array(tmp_result)
                cv2.imwrite(os.path.join(save_path, 'result'+'_'+str(epoch)+'_'+str(idx)+'.jpg'), tmp_result)
                #cv2.waitKey(0)

            print('acc: ', acc_valid, " loss", r_loss)

    return result

if __name__ == "__main__":
    train()
    print('*** training ***')
