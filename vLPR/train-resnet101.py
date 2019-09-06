import os
import sys
sys.path.insert(1, '/ghome/jinlk/lib')
from tensorboardX import SummaryWriter
import argparse
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
sys.path.insert(1, '/ghome/jinlk/jinlukang/vLPR/vLPR_POS_SEG_JOIN')
from dataset.License import License_Real_seg_pos_train, License_Real_seg_pos_val
from model.build_BiSeNet import BiSeNet
from utils import poly_lr_scheduler
from utils import reverse_one_hot, get_label_info, colour_code_segmentation, compute_global_accuracy
from metrics import runningScore


exp_name = 'baseline_train_without_night'
print(exp_name)

def isEqual(labelGT, labelP):
    compare = [1 if int(labelGT[i]) == int(labelP[i]) else 0 for i in range(7)]
    return sum(compare)


def val(args, model, dataloader):
    segmiou_cal = runningScore(n_classes=args.num_classes)
    posmiou_cal = runningScore(n_classes=args.num_char)
    model.eval()
    with torch.no_grad():
        for i, (data, seg, pos) in enumerate(dataloader):
            seg_pred, pos_pred = model(data)
            seg_pred = seg_pred.cpu().numpy()
            seg_pred = np.argmax(seg_pred, axis=1)
            seg = seg.numpy()
            segmiou_cal.update(seg, seg_pred)
            pos_pred = pos_pred.cpu().numpy()
            pos_pred = np.argmax(pos_pred, axis=1)
            pos = pos.numpy()
            posmiou_cal.update(pos, pos_pred)
    segmiou = segmiou_cal.get_scores()
    posmiou = posmiou_cal.get_scores()
    print('segmiou:{}'.format(segmiou))
    print('posmiou:{}'.format(posmiou))
    return segmiou, posmiou


def train(args, model, optimizer, criterion, dataloader_train, dataloader_val):
    writer = SummaryWriter(args.log_path)
    print(args)

    model.train()
    max_iter = len(dataloader_train)
    max_segmiou = -1
    for i, (data, seg, pos) in enumerate(dataloader_train):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=i, max_iter=max_iter)
        seg_pred, pos_pred = model(data)
        pos = pos.cuda()
        loss1 = criterion(pos_pred, pos)
        seg = seg.cuda()
        loss2 = criterion(seg_pred, seg)
        loss = loss1 + loss2
        loss = torch.mean(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('loss_step', loss.item(), i)
        print('iter:{}/{} lr:{:.5f} loss:{:.5f}'.format(i, max_iter, lr, loss.item()))

        if i % args.validation_step + 1 == args.validation_step:
            segmiou, posmiou = val(args, model, dataloader_val)
            writer.add_scalar('segmiou', segmiou, i)
            writer.add_scalar('posmiou', posmiou, i)
            if segmiou > max_segmiou:
                max_segmiou = segmiou
                torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'best.pth'))
            model.train()
    torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'last.pth'))


def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=5, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset', type=str, default="CamVid", help='Dataset you are using.')
    parser.add_argument('--crop_height', type=int, default=640, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=640, help='Width of cropped/resized input image to network')
    parser.add_argument('--train_batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--val_batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101", help='The context path model you are using.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--num_char', type=int, default=8, help='num of lincense chars (include background)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--log_path', type=str, default=None, help='tensorboard path')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')

    args = parser.parse_args(params)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)

    dataset_train = License_Real_seg_pos_train(split='train_without_night', num_epochs=args.num_epochs)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers
    )

    dataset_val = License_Real_seg_pos_val(split='val_without_night', num_epochs=1)
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.val_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers
    )

    # build model
    model = BiSeNet(args.num_classes, args.num_char, args.context_path)

    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # build optimizer

    # optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    # load pretrained model if exists
    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)
        model.module.load_state_dict(torch.load(args.pretrained_model_path), False)
        print('Done!')

    # train
    train(args, model, optimizer, criterion, dataloader_train, dataloader_val)

    # val(args, model, dataloader_val)


if __name__ == '__main__':
    params = [
        '--num_epochs', '4000',
        '--learning_rate', '0.00001',
        '--num_workers', '4',
        '--num_classes', '66',
        '--num_char', '8',
        '--train_batch_size', '256',
        '--val_batch_size', '1',
        '--context_path', 'resnet101',
        '--validation_step', '1000',
        '--log_path', '/ghome/jinlk/jinlukang/vLPR/vLPR_POS_SEG_JOIN/log/{}_pos_seg_join'.format(exp_name),
        '--pretrained_model_path', '/ghome/jinlk/zhangyesheng/ILPR/model/resnet101-init.pth',
        '--save_model_path', '/gdata/jinlk/jinlukang/vLPR/save_model/vLPR_POS_SEG_JOIN/{}_pos_seg_join'.format(exp_name)
    ]
    main(params)
