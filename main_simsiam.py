#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
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

import simsiam.loader
import simsiam.builder
import simsiam.resnet

from utils import plot_images
from meter import AverageMeter, ProgressMeter

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def arguments():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # parser.add_argument('data', metavar='DIR',
    #                     help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet50)')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=512, type=int,
                        metavar='N',
                        help='mini-batch size (default: 512), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                        metavar='LR', help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    parser.add_argument('--eval-period', default=10, type=int,
                        help='Evaluation period')
    parser.add_argument('--checkpoint-dir', default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--supervised', action='store_true',
                        help='Leverage supervised information')
    parser.add_argument('--predictor-reg', default=None, choices=['corr'],
                        help='Predictor regularization')
    parser.add_argument('--ema', type=float, default=0,
                        help='Momentum for the target encoder in SimSiam')
    parser.add_argument('--strong-aug', action='store_true',
                        help='Apply the same augmentation as in the SSL case')
    parser.add_argument('--exp-name', default=None,
                        help='Experiment name')
    parser.add_argument('--celoss-ratio', default=0., type=float,
                        help='Cross entropy loss ratio to the pipeline')

    # simsiam specific configs:
    parser.add_argument('--dim', default=2048, type=int,
                        help='feature dimension (default: 2048)')
    parser.add_argument('--pred-dim', default=512, type=int,
                        help='hidden dimension of the predictor (default: 512)')
    parser.add_argument('--fix-pred-lr', action='store_true',
                        help='Fix learning rate for the predictor')
    return parser

def main():
    args = arguments().parse_args()

    args.run_dir = os.path.join(args.checkpoint_dir,
                                args.exp_name if args.exp_name is not None else time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(args.run_dir, exist_ok=True)

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
    args.ngpus_per_node = ngpus_per_node
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


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args, **kwargs):
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
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = simsiam.builder.SimSiam(
        # models.__dict__[args.arch],
        simsiam.resnet.ResNet18,
        args.dim, args.pred_dim,
        predictor_reg=args.predictor_reg, ema=args.ema, sup_branch=args.celoss_ratio>0)
    encoder = simsiam.builder.SimSiamEncoder(model.encoder)

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    if args.distributed:
        # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            encoder = torch.nn.parallel.DistributedDataParallel(encoder)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        encoder = encoder.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
        pass
    print(model) # print model after SyncBatchNorm

    # define loss function (criterion) and optimizer
    criterion = nn.CosineSimilarity(dim=1).cuda(args.gpu)
    sup_criterion = nn.CrossEntropyLoss()

    if args.fix_pred_lr:
        if args.distributed:
            optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                            {'params': model.module.predictor.parameters(), 'fix_lr': True}]
        else:
            optim_params = [{'params': model.encoder.parameters(), 'fix_lr': False},
                            {'params': model.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()

    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    # traindir = os.path.join(args.data, 'train')
    mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    # mean, std = [0., 0., 0.], [1., 1., 1.]
    normalize = transforms.Normalize(mean=mean, std=std)

    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    augmentation = [
        transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    if args.strong_aug:
        train_transform = transforms.Compose(augmentation)
    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    if args.supervised:
        dset = datasets.CIFAR10(root='./data', train=True, transform=train_transform)
        train_dataset = simsiam.loader.SupervisedSimSiamDataset(dset)
    else:
        train_dataset = datasets.CIFAR10(root='./data', train=True,
                                         transform=simsiam.loader.TwoCropsTransform(transforms.Compose(augmentation)))

    train_knn_dataset = datasets.CIFAR10(root='./data', train=True, transform=test_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=test_transform)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_knn_sampler = torch.utils.data.distributed.DistributedSampler(train_knn_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        train_sampler = None
        train_knn_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    train_knn_loader = torch.utils.data.DataLoader(
        train_knn_dataset, batch_size=args.batch_size, shuffle=(train_knn_sampler is None),
        num_workers=4, pin_memory=True, sampler=train_knn_sampler, drop_last=False)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=(test_sampler is None),
        num_workers=4, pin_memory=True, sampler=test_sampler, drop_last=False)

    niter = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        niter = train(train_loader, model, criterion, sup_criterion, optimizer, epoch, niter, args)

        if (epoch > 0 and epoch % args.eval_period == 0) or epoch == args.epochs-1:
            # accu = test(train_knn_loader, train_knn_loader, encoder, epoch, args)
            accu = test(train_knn_loader, test_loader, encoder, epoch, args)

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, is_best=False, filename='{}/checkpoint_{:04d}_{:.2f}.pth.tar'.format(args.run_dir, epoch, accu))


def train(train_loader, model, criterion, sup_criterion, optimizer, epoch, niter, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')

    monitor = [batch_time, data_time, losses]
    if args.celoss_ratio > 0:
        sup_losses = AverageMeter('CE', ':.4f')
        accus = AverageMeter('Accu', ':.4f')
        monitor.append(sup_losses)
        monitor.append(accus)

    progress = ProgressMeter(
        len(train_loader),
        monitor,
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # plot_images(images[0], images[1])

        # compute output and loss
        p1, p2, z1, z2, logits = model(x1=images[0], x2=images[1])
        model.update(z1, z2)
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
        losses.update(loss.item(), images[0].size(0))

        # compute supervised loss and accuracy
        if args.celoss_ratio > 0:
            assert logits is not None
            labels = labels.cuda(args.gpu, non_blocking=True)
            sup_loss = sup_criterion(logits, labels)
            sup_loss = args.celoss_ratio * sup_loss
            sup_losses.update(sup_loss.item(), images[0].size(0))

            correct = logits.argmax(dim=-1).eq(labels).float().sum()
            accu = correct / images[0].size(0)
            accus.update(accu.item(), images[0].size(0))
            loss += sup_loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.regulate_predictor(niter, epoch_start=False)
        model.update_target_network_parameters()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        niter += 1

    return niter


def get_features(loader, encoder, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time])

    # switch to train mode
    encoder.eval()

    num = 0
    features, labels = [], []

    end = time.time()
    for i, (images, y) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            y = y.cuda(args.gpu, non_blocking=True)

        # compute output
        z1 = encoder(images)
        features.append(z1); labels.append(y)
        num += len(z1)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, end='\r')

    features = torch.cat(features)
    labels = torch.cat(labels)
    if args.multiprocessing_distributed:
        all_features = [torch.zeros_like(features) for _ in range(args.world_size)]
        dist.all_gather(all_features, features)
        features = torch.cat(all_features, dim=0)

        all_labels = [torch.zeros_like(labels) for _ in range(args.world_size)]
        dist.all_gather(all_labels, labels)
        labels = torch.cat(all_labels, dim=0)

    features = features / features.norm(dim=-1, keepdim=True)
    return features, labels


def test(train_knn_loader, test_loader, encoder, epoch, args, knn_k=25, knn_t=0.1):
    with torch.no_grad():
        classes = len(train_knn_loader.dataset.classes)

        # Compute features
        train_features, train_labels = get_features(train_knn_loader, encoder, args)
        test_features, test_labels = get_features(test_loader, encoder, args)

        if (args.multiprocessing_distributed and args.rank % args.ngpus_per_node != 0):
            return None

        # KNN classifier
        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        sim_matrix = torch.mm(test_features, train_features.T)
        sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
        sim_labels = torch.gather(train_labels.expand(test_features.size(0), -1), dim=-1, index=sim_indices)
        sim_weight = (sim_weight / knn_t).exp()

        # counts for each class
        one_hot_label = torch.zeros(test_features.size(0) * knn_k, classes, device=sim_labels.device)
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        pred_scores = torch.sum(one_hot_label.view(test_features.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
        predicted = pred_scores.argsort(dim=-1, descending=True)

        # Compute per-class accuracy
        match = predicted[:,0].eq(test_labels).int()

    cls_total, cls_correct = {'all': 0}, {'all': 0}
    for m, l in zip(match, test_labels):
        l, m = l.item(), m.item()
        if cls_total.get(l, None) is None:
            cls_total[l] = 0; cls_correct[l] = 0
        cls_total[l] += 1; cls_total['all'] += 1
        cls_correct[l] += m; cls_correct['all'] += m
    for cls in cls_total.keys():
        if cls == 'all': continue
        correct, total = cls_correct[cls], cls_total[cls]
        print(f'{cls}: {correct/total} ({correct}/{total})', end='; ')
    correct, total = cls_correct['all'], cls_total['all']
    accu = correct/total
    print(f'\nAVG: {accu:.4f} ({correct}/{total})')
    return accu


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


if __name__ == '__main__':
    main()
