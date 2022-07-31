# YOLO-v1-Pytorch 

hello guys, this is a yol0-v1 implementation base pytorch 

if you think it's helpful for you, plz give me a star

I will so appreciate it!

## Custom dataset
1. transfer your dataset to Pascal-VOC format
2. modify the  dataset root dirctory in voc0712.py line:28

## Backbone
- ResNet(18-152)

## Neck
- SPP

## Detection head
- Convs

## Loss function
- conf_loss_function = MSEWithLogitsLoss
- cls_loss_function = CrossEntropyLoss
- offset_xy_loss_function = BCEWithLogitsLoss
- w_h_loss_function = MSELoss

## Post process 
- NMS

## experiment
-  Pascal VOC
