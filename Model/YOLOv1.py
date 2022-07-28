import numpy as np
import torch
import torch.nn as nn
from Backbone.resnet import resnet18
from Modules.common import CBL, SPP
import Tools.loss_function as loss_function


class YOLOv1(nn.Module):
    def __init__(self, device, num_classes=20, is_train=False, input_size=None, conf_thresh=0.01, nms_thresh=0.5):
        super(YOLOv1, self).__init__()
        self.num_classes = num_classes
        self.stride = 32
        self.device = device
        self.gird_cell = self.get_grid_matrix(input_size)
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.is_train = is_train

        # backbone
        self.backbone = resnet18(pretrained=is_train)
        # channel of output
        c5 = 512

        # neck
        self.neck = nn.Sequential(
            SPP(),
            # downsample
            CBL(c5*4, c5, kernel_size=1)
        )

        # detection head
        self.convsets = nn.Sequential(
            CBL(c5, 256, kernel_size=1),
            CBL(256, 512, kernel_size=3, padding=1),
            CBL(512, 256, kernel_size=1),
            CBL(256, 512, kernel_size=3, padding=1)
        )

        # prediction head
        self.pred = nn.Conv2d(512, 1 + self.num_classes + 4, 1)

    def get_grid_matrix(self, input_size):
        """
            get matrix for all coordinate of grid center (x,y)
        """
        w, h = input_size, input_size
        ws, hs = w // self.stride, h // self.stride
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])

        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()

        grid_xy_matrix = grid_xy.view(1, hs*ws, 2).to(self.device)

        return grid_xy_matrix

    def reset_grid_matrix(self, input_size):
        """
            for reset G matrix
        """
        self.input_size = input_size
        self.grid_cell = self.get_grid_matrix(input_size)


    def decode_boxes(self, bbox_pred):
        """
            arg: bbox_pred:[x_offset, y_offset, w, h]
                 w, h = gt_w = np.log(gt_bbox_w), gt_h = np.log(gt_bbox_h)
            return: bbox:[xmin, ymin, xmax, ymax]
        """

        bbox = torch.zeros_like(bbox_pred)
        # get center_x, center_y, w, h of all bounding box
        bbox_pred[:, :, :2] = torch.sigmoid(bbox_pred[:, :, :2]) + self.grid_cell
        bbox_pred[:, :, 2:] = torch.exp(bbox_pred[:, :, 2:])

        # 将所有bbox的中心点坐标和宽高换算成x1y1x2y2形式
        bbox[:, :, 0] = bbox_pred[:, :, 0] * self.stride - bbox_pred[:, :, 2] / 2
        bbox[:, :, 1] = bbox_pred[:, :, 1] * self.stride - bbox_pred[:, :, 3] / 2
        bbox[:, :, 2] = bbox_pred[:, :, 0] * self.stride + bbox_pred[:, :, 2] / 2
        bbox[:, :, 3] = bbox_pred[:, :, 1] * self.stride + bbox_pred[:, :, 3] / 2
        
        return bbox

    def nms(self, dets, scores):
        """"
            Pure Python NMS baseline.
        """
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        

        keep = []                                             
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # 计算交集的左上角点和右下角点的坐标
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            # 计算交集的宽高
            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            # 计算交集的面积
            inter = w * h

            # 计算交并比
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # 滤除超过nms阈值的检测框
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    def postprocess(self, bboxes, scores):
        """
            bboxes: (HxW, 4), bsize = 1
            scores: (HxW, num_classes), bsize = 1
        """

        cls_inds = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), cls_inds)]
        
        # filter by threshold 
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS for each class
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        # get result
        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        return bboxes, scores, cls_inds

    def forward(self, x, targets=None):

        C_5 = self.backbone(x)

        C_5 = self.neck(C_5)

        C_5 = self.convsets(C_5)

        pred = self.pred(C_5)
        # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
        pred = pred.view(C_5.size(0), 1 + self.num_classes + 4, -1).permute(0, 2, 1)
        # confidence prediction
        # [B, H*W, 1]
        _conf_pred = pred[:, :, :1]
        # class prediction
        # [B, H*W, num_classes]
        _class_pred = pred[:, :, 1:1+self.num_classes]
        # bounding box prediction
        # [B, H*W, 4], include x_offset, y_offset, w, h
        _bbox_pred = pred[:, :, 1+self.num_classes:]

        if self.is_train:
            conf_loss, class_loss, bbox_loss, total_loss = loss_function.loss(conf_pred=_conf_pred,
                                                                      class_pred=_class_pred,
                                                                      bbox_pred=_bbox_pred,
                                                                      label=targets)
            return conf_loss, class_loss, bbox_loss, total_loss
        else:
            # test
            with torch.no_grad():
                # [B, H*W, 1] -> [H*W, 1]
                conf_pred = torch.sigmoid(conf_pred)[0]
                # [B, H*W, 4] -> [H*W, 4]
                # clamp() for limiting the value of bbox_pred in range 0 to 1
                bboxes = torch.clamp((self.decode_boxes(_bbox_pred) / self.input_size), .0, .1)
                # [B, H*W, 1] -> [H*W, 1]
                scores = (torch.softmax(_class_pred[0, :, :], dim=1) * conf_pred)
                # move var to cpu for postprocess 
                scores = scores.to('cpu').numpy()
                bboxes = bboxes.to('cpu').numpy()

                # postprocess
                bboxes, scores, class_inds = self.postprocess(bboxes, scores)
                return bboxes, scores, class_inds

