import torch
import torch.nn as nn

class MSEWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MSEWithLogitsLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        inputs = torch.clamp(torch.sigmoid(logits), min=1e-4, max=1.0 - 1e-4)

        pos_id = (targets==1.0).float()
        neg_id = (targets==0.0).float()
        pos_loss = pos_id * (inputs - targets)**2
        neg_loss = neg_id * (inputs)**2
        loss = 5.0*pos_loss + 1.0*neg_loss

        if self.reduction == 'mean':
            batch_size = logits.size(0)
            loss = torch.sum(loss) / batch_size

            return loss

        else:
            return loss


def loss(conf_pred, class_pred, bboxs_pred, label):
    '''
        total loss
    '''
    # print("conf_pred", conf_pred)
    # print("class_pred", class_pred)
    # print("bbox_pred", bbox_pred)
    # print("label", label)

    conf_loss_function = MSEWithLogitsLoss(reduction='mean')
    cls_loss_function = nn.CrossEntropyLoss(reduction='none')
    offsetxy_loss_function = nn.BCEWithLogitsLoss(reduction='none')
    w_h_loss_function = nn.MSELoss(reduction='none')

    conf_pred = conf_pred[:, :, 0]
    class_pred = class_pred.permute(0, 2, 1)
    offset_xy_pred = bboxs_pred[:, :, :2]
    w_h_pred = bboxs_pred[:, :, 2:]
    
    # ground truth
    obj_gt = label[:, :, 0]
    class_gt = label[:, :, 1].long()
    offset_xy_gt = label[:, :, 2:4]
    w_h_gt = label[:, :, 4:6]
    gt_box_scale_weight = label[:, :, 6]

    batch_size = conf_pred.size(0)

    # confidence loss
    conf_loss = conf_loss_function(conf_pred, obj_gt)
    
    # class loss
    class_loss = torch.sum(cls_loss_function(class_pred, class_gt) * obj_gt) / batch_size
    
    # bounding box loss
    offset_xy_loss = torch.sum(torch.sum(offsetxy_loss_function(offset_xy_pred, offset_xy_gt), dim=-1) * gt_box_scale_weight * obj_gt) / batch_size
    w_h_loss = torch.sum(torch.sum(w_h_loss_function(w_h_pred, w_h_gt), dim=-1) * gt_box_scale_weight * obj_gt) / batch_size
    bbox_loss = offset_xy_loss + w_h_loss

    # total loss
    total_loss = conf_loss + class_loss + bbox_loss

    return conf_loss, class_loss, bbox_loss, total_loss