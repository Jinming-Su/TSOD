import datetime
import os

import torch
import torchvision
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2

import sys
sys.path.append('../')

import joint_transforms
from config import cityscapes_train_256_path, dutstr_path
from datasets import ImageFolder
from misc import AvgMeter, check_mkdir

from fpn import FPN
from fpn_dual import FPN_dual
from torch.backends import cudnn

#from tensorboardX import SummaryWriter
#writer = SummaryWriter()

import time

cudnn.benchmark = True

torch.manual_seed(2019)
torch.cuda.set_device(0)

ckpt_path = './ckpt'

args = {
    'net': 'restnet50_dilation',
    'resize': [256, 512],  # h, w
    'max_iter': 200000,
    'iteration_of_epoch': 2975,
    'train_batch_size': 4,
    'save_interval': 5000,
    'last_iter': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'display': 10,
}

joint_transform = joint_transforms.Compose([
    #joint_transforms.Resize(args['resize']),
    #joint_transforms.RandomCrop(300),
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.RandomRotate(10)
])
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.2869, 0.3251, 0.2839], [0.1870, 0.1902, 0.1872])
])
target_transform = transforms.ToTensor()

train_set = ImageFolder(cityscapes_train_256_path, joint_transform, img_transform, target_transform)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=12, shuffle=True, drop_last=True)

criterion = nn.BCEWithLogitsLoss().cuda()
log_path = os.path.join(str(datetime.datetime.now()) + '.txt')


def main():
    net = FPN_dual().cuda().train()

    input_data = torch.rand(args['train_batch_size'], 3, args['resize'][0], args['resize'][1])
    #writer.add_graph(FPN_dual().train(), (input_data,))

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[:7]!= 'target_'],
         'lr': 0},
        {'params': [param for name, param in net.named_parameters() if name[:7]== 'target_' and name[7:12] == 'layer' and name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[:7]== 'target_' and name[7:12] == 'layer' and name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']},
	    {'params': [param for name, param in net.named_parameters() if name[:7]== 'target_' and name[7:12] != 'layer' and name[-4:] == 'bias'],
         'lr': 20 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[:7]== 'target_' and name[7:12] != 'layer' and name[-4:] != 'bias'],
         'lr': 10 *args['lr'], 'weight_decay': args['weight_decay']},
    ], momentum=args['momentum'])

    if len(args['snapshot']) > 0:
        print('training resumes from ' + args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, args['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, args['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)


def train(net, optimizer):
    curr_iter = args['last_iter']
    while True:
        total_loss_record, loss1_record, loss2_record, loss3_record, loss4_record, loss5_record = \
            AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

        start_time = time.time()

        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['max_iter']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['max_iter']
                                                            ) ** args['lr_decay']

            inputs, labels, labels_b, labels_i = data

            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()
            labels_b = Variable(labels_b).cuda()
            labels_i = Variable(labels_i).cuda()

            optimizer.zero_grad()
            outputs1, outputs2, outputs3, outputs4, outputs5, \
                outputs1_b, outputs2_b, outputs3_b, outputs4_b, outputs5_b, \
                outputs1_i, outputs2_i, outputs3_i, outputs4_i, outputs5_i = net(inputs)
            loss1 = criterion(outputs1, labels)
            loss2 = criterion(outputs2, labels)
            loss3 = criterion(outputs3, labels)
            loss4 = criterion(outputs4, labels)
            loss5 = criterion(outputs5, labels)

            loss1_b = criterion(outputs1_b, labels_b)
            loss2_b = criterion(outputs2_b, labels_b)
            loss3_b = criterion(outputs3_b, labels_b)
            loss4_b = criterion(outputs4_b, labels_b)
            loss5_b = criterion(outputs5_b, labels_b)

            loss1_i = criterion(outputs1_i, labels_i)
            loss2_i = criterion(outputs2_i, labels_i)
            loss3_i = criterion(outputs3_i, labels_i)
            loss4_i = criterion(outputs4_i, labels_i)
            loss5_i = criterion(outputs5_i, labels_i)

            total_loss = loss1 + loss2 + loss3 + loss4 + loss5 +\
                            loss1_b + loss2_b + loss3_b + loss4_b + loss5_b +\
                            loss1_i + loss2_i + loss3_i + loss4_i + loss5_i

            #writer.add_image('image', inputs[0].squeeze())
            #writer.add_image('output', nn.functional.sigmoid(outputs1[0]))
            #writer.add_image('label', labels[0])

            total_loss.backward()
            optimizer.step()

            total_loss_record.update(total_loss.item(), batch_size)
            loss1_record.update(loss1.item(), batch_size)
            loss2_record.update(loss2.item(), batch_size)
            loss3_record.update(loss3.item(), batch_size)
            loss4_record.update(loss4.item(), batch_size)
            loss5_record.update(loss5.item(), batch_size)

            curr_iter += 1

            if curr_iter % args['display'] == 0:
                #writer.add_scalar('loss', total_loss, global_step=i)
                total_time = time.time()-start_time
                rest_time = (total_time * 1.0 / args['display'])*(args['max_iter'] - curr_iter)
                start_time = time.time()

                curr_epoch = curr_iter * args['train_batch_size'] / args['iteration_of_epoch'] + 1

                log = '%s : e %d | iter %d | total l %.3f|tl %.3f | l0 %.3f | l1 %.3f | l2 %.3f | l3 %.3f | l4 %.3f | lr %.5f | t/s %.2f  rest/s %d:%d:%d' % \
                      (str(datetime.datetime.now()), curr_epoch, curr_iter, total_loss_record.avg, total_loss, \
                       loss1, loss2, loss3, loss4, loss5, optimizer.param_groups[1]['lr'], total_time, \
                       int(rest_time/3600), int(rest_time%3600/60), int(rest_time%60))

                print(log)
                open(log_path, 'a').write(log + '\n')


            if curr_iter % args['save_interval'] == 0:
                torch.save(net.state_dict(), os.path.join(ckpt_path, '%d.pth' % curr_iter))
                torch.save(optimizer.state_dict(),
                           os.path.join(ckpt_path, '%d_optim.pth' % curr_iter))
            if curr_iter >=  args['max_iter']:
                return 

if __name__ == '__main__':
    main()
