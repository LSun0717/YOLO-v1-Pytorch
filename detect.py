import time
import cv2
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from Dataset_utils import BaseTransform
from Dataset_utils.voc0712 import VOC_CLASSES, VOC_ROOT, VOCDetection
import Dataset_utils.cocodataset as COCO

def parse_args():
    parser = argparse.ArgumentParser(description="detection with yolo")
    parser.add_argument('--version', default='yolov1', help="version of yolo")
    parser.add_argument('--data_source', default="voc", help="data source")
    parser.add_argument("--input_size", default=416, type=int, help="size of input image")
    parser.add_argument("--weight", default="Experiments\\result_experiment\\32_ms\yolov1_150.pth", type=str, help="model weight directory")
    parser.add_argument("--conf_thresh", default=0.10, type=float, help="threshold of confidence")
    parser.add_argument("--nms_thresh", default=0.50, type=float, help="threshold of Non-Maximum Suppression")
    parser.add_argument("--vis_thresh", default=0.30, type=float, help="threshold of result visiualization ")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    return parser.parse_args()

def vis(img, bboxes, scores, cls_inds, thresh, class_colors, class_names, class_indexs=None, dataset='voc'):
    '''
        Visiualize the bounding box and class_name
    '''
    if dataset == 'voc':
        for i, box in enumerate(bboxes):
            cls_indx = cls_inds[i]
            xmin, ymin, xmax, ymax = box
            if scores[i] > thresh:
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_colors[int(cls_indx)], 1)
                cv2.rectangle(img, (int(xmin), int(abs(ymin)-20)), (int(xmax), int(ymin)), class_colors[int(cls_indx)], -1)
                mess = '%s' % (class_names[int(cls_indx)])
                cv2.putText(img, mess, (int(xmin), int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    elif dataset == 'coco-val' and class_indexs is not None:
        for i, box in enumerate(bboxes):
            cls_indx = cls_inds[i]
            xmin, ymin, xmax, ymax = box
            if scores[i] > thresh:
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_colors[int(cls_indx)], 1)
                cv2.rectangle(img, (int(xmin), int(abs(ymin)-20)), (int(xmax), int(ymin)), class_colors[int(cls_indx)], -1)
                cls_id = class_indexs[int(cls_indx)]
                cls_name = class_names[cls_id]
                # mess = '%s: %.3f' % (cls_name, scores[i])
                mess = '%s' % (cls_name)
                cv2.putText(img, mess, (int(xmin), int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    return img

def detect(model, device, testset, transform, thresh, class_colors=None, class_names=None, class_index=None, class_indexs=None, dataset='voc'):

    num_imgs = len(dataset)
    for index in range(num_imgs):
        print("Testing img: {} / {}".format(index+1, num_imgs))
        img, _ = testset.pull_image(index)
        h, w, _ = img.shape
        # basetransform
        x = torch.from_numpy(transform(img)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
        x = x.unsqueeze(0).to(device)
        t0 = time.time()

        bboxs, scores, cls_inds = model(x)
        # normalized bounding box to real world bounding box  
        scale = np.array([[w, h, w, h]])
        bboxs *= scale

        img_processed = vis(img, bboxs, scores, cls_inds, thresh, class_colors, class_names, class_index, dataset)
        cv2.imshow('detection', img_processed)
        cv2.waitKey(0)
        # print('Saving the' + str(index) + '-th image ...')
        # cv2.imwrite('test_images/' + args.dataset+ '3/' + str(index).zfill(6) +'.jpg', img)

if __name__ == "__main__":

    args = parse_args()
    # choose device for evaluate
    if args.cuda:
        print("use cuda")
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    input_size = args.input_size

    # use voc
    if args.data_source == "voc":
        print("test on voc")
        class_names = VOC_CLASSES
        class_indexs = None
        num_classes = 20
        dataset = VOCDetection(root=VOC_ROOT, img_size=input_size, image_sets=[('2007', 'test')], transform=None)
    # use coco
    elif args.data_source == "coco-val":
        print("test on coco-val")        
        class_names = COCO.coco_class_labels
        class_indexs = COCO.coco_class_index 
        num_classes = 80

        dataset = COCO.COCODataset( data_dir=COCO.coco_root, 
                                    json_file="instance_val2017.json", 
                                    name="val2017", 
                                    img_size=input_size
                                    )

    # for visiualization of each class bounding box 
    class_colors = [(np.random.randint(255),np.random.randint(255),np.random.randint(255)) for _ in range(num_classes)]

    if args.version == "yolov1":
        from Model.YOLOv1 import YOLOv1
        model = YOLOv1(device=device, input_size=input_size, num_classes=num_classes, is_train=False)
    else:
        print("Unknow Model-----------")
        exit()

    model.load_state_dict(torch.load(args.weight, map_location=device))
    model.to(device).eval()
    print('finished loading model')

    # begin test
    detect( model=model, 
            device=device, 
            testset=dataset,
            transform=BaseTransform(input_size),
            thresh=args.vis_thresh,
            class_colors=class_colors,
            class_names=class_names,
            class_indexs=class_indexs,
            dataset=args.data_source
        )



