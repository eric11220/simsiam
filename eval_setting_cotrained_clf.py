#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import glob
import math
import numpy as np
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import simsiam.resnet
from main_simsiam import arguments

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = arguments()
parser.add_argument('--pretrained', default='', type=str,
                    help='path to simsiam pretrained checkpoint')
parser.add_argument('--epoch-per-task', default=100, type=int,
                    help='how many epochs were each task trained for')
parser.add_argument('--nclass', default=10, type=int)
parser.add_argument('--test-mode', deafult='knn', choices=['knn', 'orig_clf', 'lincls'])

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def load_state_dict(path):
    checkpoint = torch.load(path, map_location="cpu")

    # rename moco pre-trained keys
    state_dict = checkpoint['state_dict']

    state_dict['encoder.fc.weight'] = state_dict['cls.weight']
    state_dict['encoder.fc.bias'] = state_dict['cls.bias']

    for k in list(state_dict.keys()):
        if k.startswith('encoder'):
            state_dict[k[len("encoder."):]] = state_dict[k]
        else:
            del state_dict[k]
    return state_dict


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    # Data loading code
    mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    normalize = transforms.Normalize(mean=mean, std=std)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root='./data', train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
        ])),
        batch_size=256, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = simsiam.resnet.ResNet18(10).cuda(args.gpu)

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            state_dict = load_state_dict(args.pretrained)
            model.load_state_dict(state_dict, strict=False)
            print("=> loaded pre-trained model '{}'".format(args.pretrained))

            # define loss function (criterion) and optimizer
            criterion = nn.CrossEntropyLoss().cuda(args.gpu)
            validate(val_loader, model, criterion, args)

        elif os.path.isdir(args.pretrained):
            ckpt_paths = [
                glob.glob('{}/checkpoint_{:04d}_*.pth.tar'.format(args.pretrained, task * args.epoch_per_task-1))[0] \
                for task in range(1, args.ntask+1)
            ]
            for path in ckpt_paths:
                if not os.path.exists(path):
                    raise Exception(f'{path} does not exist')

            accu_mat = np.zeros((args.ntask, args.nclass))
            for task, path in enumerate(ckpt_paths):
                state_dict = load_state_dict(path)
                model.load_state_dict(state_dict, strict=False)
                print("=> loaded pre-trained model '{}'".format(args.pretrained))

                # define loss function (criterion) and optimizer
                criterion = nn.CrossEntropyLoss().cuda(args.gpu)
                _, class_correct, class_total = validate(val_loader, model, criterion, args)
                for cls in range(args.nclass):
                    accu_mat[task][cls] = class_correct[cls] / class_total[cls]

            task_mat = np.zeros((args.ntask, args.ntask))
            class_groups =  np.split(np.arange(args.nclass), args.ntask)
            for task, grp in enumerate(class_groups):
                task_mat[:, task] = np.mean(accu_mat[:, grp], axis=-1)

            forgetting = task_mat[np.arange(args.ntask), np.arange(args.ntask)] - task_mat[-1]
            print(task_mat)
            print(forgetting, np.mean(forgetting))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    corrects, totals = {}, {}
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5), corrects=corrects, totals=totals)
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        for cls in sorted(list(corrects.keys())):
            print(f'{cls}: {corrects[cls]/totals[cls]}')

    return top1.avg, corrects, totals


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,), corrects=None, totals=None):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        if corrects is not None:
            classes = torch.unique(target)
            for cls in classes.cpu().numpy():
                if corrects.get(cls, None) is None:
                    corrects[cls], totals[cls] = 0, 0
                ind = target == cls
                totals[cls] += torch.sum(ind.int()).item()
                corrects[cls] += torch.sum(correct[0, ind].int()).item()

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
