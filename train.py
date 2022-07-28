import argparse
import os
import random
import time
from cv2 import transform
import numpy as np
from sklearn.utils import shuffle
import torch
import torch.backends.cudnn as cudnn
from torch.utils import data
import torch.optim as optim

from Augmentation.augmentations import SSDAugmentation
import Config.train_config
import Tools.get_label
import Eval_utils.vocapi_evaluator
from Dataset_utils.voc0712 import VOC_ROOT, VOCDetection
from Dataset_utils.voc0712 import VOC_CLASSES
from Dataset_utils import BaseTransform
from Dataset_utils import detection_collate


def parse_args():
    parser = argparse.ArgumentParser(description="yolov1")
    parser.add_argument('--version', default='yolov1', help="version of model")
    parser.add_argument('--dataset', default='voc', help="dataset for train")
    parser.add_argument('--batch_size', default=32, type=int, help="batch size for train")
    parser.add_argument('--lr', default=1e-3, type=float, help="learing rate for train")
    parser.add_argument('--multi_scale', action='store_true', default=False, help='use multi-scale trick') 
    parser.add_argument("--no_warmup", action="store_true", default=False, help="select whether to use warmup learining policy")
    parser.add_argument("--wp_epoch", default=1, type=int, help="the upper bound of warm up")
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch to train')
    parser.add_argument('--resume', default=None, type=str, help='keep training')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--eval_epoch', type=int,default=10, help='interval between evaluations')
    parser.add_argument('--cuda', action='store_true', default=False, help='use cuda.')
    parser.add_argument('--tfboard', action='store_true', default=False, help='use tensorboard')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode where only one image is trained')
    parser.add_argument('--save_folder', default='weights/', type=str, help='save weight into save_folder')
    return parser.parse_args()

def set_lr(optimizer, lr):
    '''
        for set learning rate
    '''
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train():
    args = parse_args()

    print("Setting Argments: ", args)
    print("---------------------------------------")

    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)
    # use cuda
    if args.cuda:
        cudnn.benchmark = True
        device = torch.device("cuda")
        print("use {}".format(device))
    else:
        device = torch.device("cpu")

    # use multi scale training
    if args.multi_scale:
        print('use the multi-scale trick ...')
        train_size = 640
        val_size = 416
    else:
        train_size = 416
        val_size = 416

    cfg = Config.train_config.train_cfg

    # choose dataset
    if args.dataset == "voc":
        dataset_dir = VOC_ROOT
        num_classes = 20
        dataset = VOCDetection(root=dataset_dir,transform=SSDAugmentation(train_size))
        evaluator = Eval_utils.vocapi_evaluator.VOCAPIEvaluator(data_root=dataset_dir,
                                                                img_size=val_size,
                                                                device=device,
                                                                transform=BaseTransform(val_size),
                                                                labelmap=VOC_CLASSES
                                                                )
    # COCO Dataset
    # elif args.dataset == 'coco':
    #     # 加载COCO数据集
    #     data_dir = coco_root
    #     num_classes = 80
    #     dataset = COCODataset(
    #                 data_dir=data_dir,
    #                 img_size=train_size,
    #                 transform=SSDAugmentation(train_size),
    #                 debug=args.debug
    #                 )

    #     evaluator = COCOAPIEvaluator(
    #                     data_dir=data_dir,
    #                     img_size=val_size,
    #                     device=device,
    #                     transform=BaseTransform(val_size)
    #                     )

    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)

    print("training model on: {}".format(dataset.name))
    print("the dataset size : {}".format(len(dataset)))
    print("----------------------------------------")

    # construc dataloader
    dataloader = data.DataLoader(dataset,
                                batch_size = args.batch_size,
                                shuffle=True,
                                collate_fn = detection_collate,
                                num_workers = args.num_workers,
                                pin_memory=True
                                )
    # construc model
    if args.version == "yolov1":
        from Model.YOLOv1 import YOLOv1
        yolov1 = YOLOv1(device=device, num_classes=num_classes, is_train=True, input_size=train_size)
        print("train yolo on {} dataset".format(args.dataset))

    else:
        print("this code is just for yolo")
        exit()

    model = yolov1
    model.to(device).train()

    # use tensor board
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        log_path = os.path.join('log/coco/', args.version, c_time)
        os.makedirs(log_path, exist_ok=True)

        writer = SummaryWriter(log_path)    
    if args.resume is not None:
        print("resume training model {}", args.resume)
        model.load_state_dict(torch.load(args.resume, map_location=device))

    # construct optimizer
    base_lr = args.lr
    tmp_lr = base_lr
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay
                          )

    max_epoch = cfg['max_epoch']
    epoch_size = len(dataset) // args.batch_size

    # begin training
    t0 = time.time()
    for epoch in range(args.start_epoch, max_epoch):

        # Ladder learning rate attenuation
        if epoch in cfg['lr_epoch']:
            tmp_lr = tmp_lr * 0.1
            set_lr(optimizer, tmp_lr)

        for iter_i, (images, targets) in enumerate(dataloader):
            # targets.shape = batch_size
            # every element in targets is [bbox, class_index]
            # warm up training policy
            if not args.no_warmup:
                if epoch < args.wp_epoch:
                    tmp_lr = base_lr * pow((iter_i + epoch * epoch_size)*1. / (args.wp_epoch * epoch_size), 4)
                    set_lr(optimizer, tmp_lr)

                elif epoch == args.wp_epoch and iter_i == 0:
                    tmp_lr = base_lr
                    set_lr(optimizer, tmp_lr) 

            # multi scale training
            if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
                # random size
                train_size = random.randint(10, 19) * 32
                model.reset_grid_matrix(train_size)

            if args.multi_scale:
                # interpolate
                images = torch.nn.functional.interpolate(images, size=train_size, mode='bilinear', align_corners=False)

            # generate label for train 
            targets = [label.tolist() for label in targets]
            
            targets = Tools.get_label.get_groundtruth(input_size=train_size, stride=yolov1.stride, label_lists=targets)
            # targets.shape = [batch_size, -1, 7]
            # every element in it is [objectness, class_index, bbox, weight]

            # to device
            images = images.to(device)
            targets = targets.to(device)

            # inference and loss computation
            conf_loss, class_loss, bbox_loss, total_loss = model(images, targets=targets)

            # back porpagation
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # training information
            if iter_i % 10 == 0:
                if args.tfboard:
                    # viz loss
                    writer.add_scalar('obj loss', conf_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('cls loss', class_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('box loss', bbox_loss.item(), iter_i + epoch * epoch_size)
                
                t1 = time.time()
                print('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                    '[Loss: obj %.2f || cls %.2f || bbox %.2f || total %.2f || size %d || time: %.2f]'
                        % (epoch+1, max_epoch, iter_i, epoch_size, tmp_lr,
                            conf_loss.item(), 
                            class_loss.item(), 
                            bbox_loss.item(), 
                            total_loss.item(), 
                            train_size, 
                            t1-t0),
                        flush=True)

                t0 = time.time()

        # evaluation
        if (epoch + 1) % args.eval_epoch == 0:
            model.is_train = False
            model.reset_grid_matrix(val_size)
            model.eval()

            # evaluate
            evaluator.evaluate(model)

            # convert to training mode.
            model.is_train = True
            model.reset_grid_matrix(train_size)
            model.train()

        # save model
        if (epoch + 1) % 10 == 0:
            print('Saving state, epoch:', epoch + 1)
            torch.save(model.state_dict(), os.path.join(path_to_save, args.version + '_' + repr(epoch + 1) + '.pth'))  
            
if __name__ == "__main__":
    train()