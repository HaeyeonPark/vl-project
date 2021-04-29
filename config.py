import os
import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision.models as models
import torch.backends.cudnn as cudnn
import random
import numpy as np
import logging

from datasets.pedes import CuhkPedes
from models.model import Model
from utils import directory
from utils.metric import Loss

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def data_config(image_dir, anno_dir, batch_size, split, max_length, transform, vocab_path='', min_word_count=0, cap_transform=None, vis=False, rand_sample=False, dist=False):
    
    data_split = CuhkPedes(image_dir, anno_dir, split, max_length, transform, \
                            vocab_path=vocab_path, min_word_count=min_word_count, cap_transform=cap_transform, rand_sample=rand_sample)
    
    if dist == True:
        ddp_sampler = DistributedSampler(data_split)
        train_shuffle = False
    else:
        ddp_sampler = None
        train_shuffle = True

    if split == 'train':
        loader = data.DataLoader(data_split, batch_size, shuffle=train_shuffle, num_workers=4, drop_last=True, sampler=ddp_sampler)
    else:
        loader = data.DataLoader(data_split, batch_size, shuffle=False, num_workers=4, drop_last=True, sampler=ddp_sampler)    
        if vis == True:
            return loader, data_split.test_images
    return loader

def get_image_unique(image_dir, anno_dir, batch_size, split, max_length, transform):
    return CuhkPedes(image_dir, anno_dir, split, max_length, transform).unique

def network_config(args, split='train', param=None, resume=False, model_path=None, param2=None):
    network = Model(args).cuda()
    if args.distributed:
        network = DDP(network, device_ids=[args.local_rank], output_device=args.local_rank)
    #network = nn.DataParallel(network).cuda()
    cudnn.benchmark = True
    args.start_epoch = 0

    # process network params
    if resume or split == 'test':
        directory.check_file(model_path, 'model_file')
        checkpoint = torch.load(model_path)
        args.start_epoch = checkpoint['epoch'] + 1
        if args.distributed:
            network.module.load_state_dict(checkpoint['network'])
        else:
            network.load_state_dict(checkpoint['network'])
        print('==> Loading checkpoint "{}"'.format(model_path))
    else:
        # pretrained
        if model_path is not None:
            print('==> Loading from pretrained models')
            network_dict = network.state_dict()
            # process keyword of pretrained model
            cnn_pretrained = torch.load(model_path)
            network_keys = network_dict.keys()
            if args.distributed:
                prefix = 'module.image_model.'
            else:
                prefix = 'image_model.'
            update_pretrained_dict = {}
            for k,v in cnn_pretrained.items():
                if prefix+k in network_keys:
                    update_pretrained_dict[prefix+k] = v
                if prefix+'branch2_'+k in network_keys:
                    update_pretrained_dict[prefix+'branch2_'+k] = v
                if prefix+'branch3_'+k in network_keys:
                    update_pretrained_dict[prefix+'branch3_'+k] = v
                if prefix+k not in network_keys and prefix+'branch2_'+k not in network_keys and prefix+'branch3_'+k not in network_keys:
                    print("warning: " + k + ' not load')
            network_dict.update(update_pretrained_dict)
            network.load_state_dict(network_dict)
                    
           
    # process optimizer params
    if split == 'test':
        optimizer = None
    else:
        # optimizer
        # different params for different part
        if args.distributed:
            cnn_params = list(map(id, network.module.image_model.parameters()))
            lang_params = list(map(id, network.module.language_model.parameters()))
        else:
            cnn_params = list(map(id, network.image_model.parameters()))
            lang_params = list(map(id, network.language_model.parameters()))
        cnn_params = cnn_params + lang_params
        other_params = filter(lambda p: id(p) not in cnn_params, network.parameters())
        other_params = list(other_params)

        if param is not None:
            other_params.extend(list(param))

        if param2 is not None:
            other_params.extend(list(param2))

        if args.distributed:
            param_groups = [{'params':other_params},
            {'params':network.module.image_model.parameters(), 'weight_decay':args.wd, 'lr':args.lr/10},
            {'params':network.module.language_model.parameters(), 'lr':args.lr/10}]
        else:
            param_groups = [{'params':other_params},
            {'params':network.image_model.parameters(), 'weight_decay':args.wd, 'lr':args.lr/10},
            {'params':network.language_model.parameters(), 'lr':args.lr/10}]
        optimizer = torch.optim.Adam(
            param_groups,
            lr = args.lr, betas=(args.adam_alpha, args.adam_beta), eps=args.epsilon)
        if resume:
            optimizer.load_state_dict(checkpoint['optimizer'])

    print('Total params: %2.fM' % (sum(p.numel() for p in network.parameters()) / 1000000.0))
    # seed
    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    return network, optimizer

def loss_config(args):
    compute_loss = Loss(args).cuda()
    if args.distributed:
        compute_loss = DDP(compute_loss, device_ids=[args.local_rank], output_device=args.local_rank)
    if args.resume:
        checkpoint = torch.load(args.model_path) 
        if args.distributed:
            compute_loss.module.load_state_dict(checkpoint['loss'])
        else:
            compute_loss.load_state_dict(checkpoint['loss'])
    return compute_loss



def log_config(args, ca):
    filename = args.log_dir +'/' + ca + '.log'
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(handler)
    logging.info(args)


def dir_config(args):
    if not os.path.exists(args.image_dir):
        raise ValueError('Supply the dataset directory with --image_dir')
    if not os.path.exists(args.anno_dir):
        raise ValueError('Supply the anno file with --anno_dir')
    directory.makedir(args.log_dir)
    # save checkpoint
    directory.makedir(args.checkpoint_dir)


def lr_scheduler(optimizer, args):
    if '_' in args.epochs_decay:
        epochs_list = args.epochs_decay.split('_')
        epochs_list = [int(e) for e in epochs_list]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, epochs_list)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(args.epochs_decay))
    return scheduler
