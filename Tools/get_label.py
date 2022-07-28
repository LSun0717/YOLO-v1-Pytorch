import numpy as np
import torch
import torch.nn as nn


def generate_offsetxy_wh(gt_label, w, h, s):
    '''
        generate the label for training from gt_label 
        input:
            gt_label[]

            orginal size
            stride
        return:
            label for training
    '''
    xmin, ymin, xmax, ymax = gt_label[:-1]
    gt_class = int(gt_label[-1])

    # get center coordinate and unnormallized w h of bbox
    gt_center_x = (xmax + xmin) / 2 * w
    gt_center_y = (ymax + ymin) / 2 * h
    gt_bbox_w = (xmax - xmin) * w
    gt_bbox_h = (ymax - ymin) * h

    if gt_bbox_w < 1e-4 or gt_bbox_h < 1e-4:
        return False
    # get the gridcell coords according to bbox's center-coord
    gt_center_x_s = gt_center_x / s
    gt_center_y_s = gt_center_y / s
    gt_gridcell_x = int(gt_center_x_s)
    gt_gridcell_y = int(gt_center_y_s)

    gt_offset_x = gt_center_x_s - gt_gridcell_x
    gt_offset_y = gt_center_y_s - gt_gridcell_y

    gt_w = np.log(gt_bbox_w)
    gt_h = np.log(gt_bbox_h)

    weight = 2.0 - (gt_bbox_w / w) * (gt_bbox_h / h)

    return gt_class, gt_gridcell_x, gt_gridcell_y, gt_offset_x, gt_offset_y, gt_w, gt_h, weight

def get_groundtruth(input_size, stride, label_lists=[]):
    '''
        return tensor for training
        label_lists:  [img1_label, img2_label, img3_label, ...]
        img1_label:   [obj1, obj2, obj3, ...]
        obj:          [class_index, x, y, h, w]
    '''
    batch_size = len(label_lists)

    w = input_size
    h = input_size
    ws = w // stride
    hs = h // stride

    gt_tensor = np.zeros([batch_size, hs, ws, 1+1+4+1])

    for index in range(batch_size):
        for gt_label in label_lists[index]:

            result = generate_offsetxy_wh(gt_label, w, h, stride)
            
            if result:
                gt_class, gt_gridcell_x, gt_gridcell_y, gt_offset_x, gt_offset_y, gt_bbox_w, gt_bbox_h, weight = result

                if gt_gridcell_x < gt_tensor.shape[2] and gt_gridcell_y < gt_tensor.shape[1]:
                    gt_tensor[index, gt_gridcell_y, gt_gridcell_x, 0] = 1.0
                    gt_tensor[index, gt_gridcell_y, gt_gridcell_x, 1] = gt_class
                    gt_tensor[index, gt_gridcell_y, gt_gridcell_x, 2:6] = np.array([gt_offset_x, gt_offset_y, gt_bbox_w, gt_bbox_h]) 
                    gt_tensor[index, gt_gridcell_y, gt_gridcell_x, 6] = weight

    gt_tensor = gt_tensor.reshape(batch_size, -1, 1+1+4+1)

    return torch.from_numpy(gt_tensor).float()
    
