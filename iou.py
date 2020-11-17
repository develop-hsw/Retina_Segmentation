import numpy as np
import cv2
import os

def iou_evlt(valid_mask, result):
    for idx in range(0, 10):
        result = np.array(result, dtype=np.uint8)
        valid_mask = np.array(valid_mask, dtype=np.uint8)

        #union = cv2.bitwise_or(result[idx], valid_mask[idx]) * 50
        intersection = cv2.bitwise_and(result[idx], valid_mask[idx])
        iou_valid = np.sum(valid_mask[idx])
        #iou_valid = np.sum(union)
        iou_inter = np.sum(intersection)
        iou_valid.astype(np.uint8)
        iou_inter.astype(np.uint8)

        ious = iou_inter / iou_valid

        return intersection, ious

if __name__ == "__main__":
    print("*** iou ***")