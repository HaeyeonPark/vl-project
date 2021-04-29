import os
import sys
import time
import shutil
import logging
import gc
import torch
import torchvision.transforms as transforms
from utils.metric import AverageMeter, compute_topk
from test_config import config
from config import data_config, network_config, get_image_unique
import numpy as np
import math
import re
import glob

import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP


def test(data_loader, network, args, unique_image):
    batch_time = AverageMeter()

    # switch to evaluate mode
    network.eval()
    ''' 
    global_img_feat_bank = []
    global_text_feat_bank = []

    local_img_query_bank = []
    local_img_value_bank = []

    local_text_key_bank = []
    local_text_value_bank = []

    labels_bank_i = []
    labels_bank_t = []
    length_bank = []
    '''
    max_size = 64 * len(data_loader)
    global_img_feat_bank = torch.zeros((max_size, args.feature_size)).cuda()
    global_text_feat_bank = torch.zeros((max_size*2, args.feature_size)).cuda()

    local_img_query_bank = torch.zeros((max_size, args.part2 + args.part3 + 1, args.feature_size)).cuda()
    local_img_value_bank = torch.zeros((max_size, args.part2 + args.part3 + 1, args.feature_size)).cuda()

    local_text_key_bank = torch.zeros((max_size*2, 98 + 2 + 1, args.feature_size)).cuda()
    local_text_value_bank = torch.zeros((max_size*2, 98 + 2 + 1, args.feature_size)).cuda()

    labels_bank_i = torch.zeros(max_size).cuda()
    labels_bank_t = torch.zeros(max_size*2).cuda()
    length_bank = torch.zeros(max_size*2, dtype=torch.long).cuda()
    index_i = 0
    index_t = 0


    with torch.no_grad():
        end = time.time()
        for images, caption, labels in data_loader:
            captions = caption[0] + caption[1]
            sep_captions = []
            n_sep = 2
            for i, c in enumerate(captions):
                #c = re.split(r'[;,!?.]', c)
                c = list(filter(None, re.split(r'[;,!?.]', c)))
                # fix: only consider first two subsentence
                if len(c) > n_sep or len(c) == n_sep:
                    sep_captions = sep_captions + c[0:n_sep]
                else:
                    pad_length = n_sep - len(c)
                    padding = ["[PAD]" for j in range(pad_length)]
                    sep_captions = sep_captions + c + padding   

            if isinstance(network, DDP):
                tokens, segments, input_masks, caption_length = network.module.language_model.pre_process(captions)
                sep_tokens, sep_segments, sep_input_masks, sep_caption_length = network.module.language_model.pre_process(sep_captions)
            else:
                tokens, segments, input_masks, caption_length = network.language_model.pre_process(captions)
                sep_tokens, sep_segments, sep_input_masks, sep_caption_length = network.language_model.pre_process(sep_captions)

            tokens = tokens.cuda()
            segments = segments.cuda()
            input_masks = input_masks.cuda()
            caption_length = caption_length.cuda()

            sep_tokens = sep_tokens.cuda()
            sep_segments = sep_segments.cuda()
            sep_input_masks = sep_input_masks.cuda()
            
            images = images.cuda()
            labels = labels.cuda()
            interval_i = images.shape[0]
            interval_t = 2*interval_i

            p2 = [i for i in range(args.part2)]
            p3 = [i for i in range(args.part3)]

            global_img_feat, global_text_feat, local_img_query, local_img_value, local_text_key, local_text_value = network(images, tokens, segments, input_masks, sep_tokens, sep_segments, sep_input_masks, n_sep, p2, p3)
            '''
            global_img_feat_bank.append(global_img_feat)
            global_text_feat_bank.append(global_text_feat)

            local_img_query_bank.append(local_img_query)
            local_img_value_bank.append(local_img_value) 

            local_text_key_bank.append(local_text_key)
            local_text_value_bank.append(local_text_value) 

            labels_bank_i.append(labels)
            labels_bank_t.append(torch.cat((labels, labels))) 
            length_bank.append(caption_length) 
            batch_time.update(time.time() - end)
            end = time.time()
            '''
            global_img_feat_bank[index_i: index_i + interval_i] = global_img_feat
            global_text_feat_bank[index_t: index_t + interval_t] = global_text_feat
            local_img_query_bank[index_i: index_i + interval_i, :, :] = local_img_query
            local_img_value_bank[index_i: index_i + interval_i, :, :] = local_img_value
            local_text_key_bank[index_t: index_t + interval_t, :, :] = local_text_key
            local_text_value_bank[index_t: index_t + interval_t, :, :] = local_text_value
            labels_bank_i[index_i:index_i + interval_i] = labels
            labels_bank_t[index_t:index_t + interval_t] = torch.cat((labels, labels))
            length_bank[index_t:index_t + interval_t] = caption_length
            batch_time.update(time.time() - end)
            end = time.time()
            index_i = index_i + interval_i
            index_t = index_t + interval_t

        '''
        global_img_feat_bank = torch.cat(global_img_feat_bank, dim=0)
        global_text_feat_bank = torch.cat(global_text_feat_bank, dim=0)
        local_img_query_bank = torch.cat(local_img_query_bank, dim=0)
        local_img_value_bank = torch.cat(local_img_value_bank, dim=0)
        local_text_key_bank = torch.cat(local_text_key_bank, dim=0)
        local_text_value_bank = torch.cat(local_text_value_bank, dim=0)
        labels_bank_i = torch.cat(labels_bank_i, dim=0)
        labels_bank_t = torch.cat(labels_bank_t, dim=0)
        length_bank = torch.cat(length_bank, dim=0)
        #unique_image = torch.tensor(unique_image) == 1
        '''
        global_img_feat_bank = global_img_feat_bank[:index_i]
        global_text_feat_bank = global_text_feat_bank[:index_t]
        local_img_query_bank = local_img_query_bank[:index_i]
        local_img_value_bank = local_img_value_bank[:index_i]
        local_text_key_bank = local_text_key_bank[:index_t]
        local_text_value_bank = local_text_value_bank[:index_t]
        labels_bank_i = labels_bank_i[:index_i]
        labels_bank_t = labels_bank_t[:index_t]
        length_bank = length_bank[:index_t]
        unique_image = torch.tensor(unique_image) == 1

        '''
        if args.distributed:
            num_dev = torch.cuda.device_count()

            #######
            global_img_feat_bank_l = [torch.zeros_like(global_img_feat_bank) for k in range(num_dev)]
            #print('***', args.local_rank, global_img_feat_bank[0][0])
            dist.all_gather(global_img_feat_bank_l, global_img_feat_bank)
            global_img_feat_bank_l = torch.cat(global_img_feat_bank_l, dim=0)
            print(global_img_feat_bank_l[0][0], global_img_feat_bank_l[1000][0])

            labels_bank_i_l = [torch.zeros_like(labels_bank_i) for k in range(num_dev)]
            print('@@@@', args.local_rank, ':', labels_bank_i[0])
            dist.all_gather(labels_bank_i_l, labels_bank_i)
            labels_bank_i_l = torch.cat(labels_bank_i_l, dim=0)
            print('gather: ', labels_bank_i_l[0:5])
            print('gather: ', labels_bank_i_l[768:768+5])
            print('gather: ', labels_bank_i_l[1536:1536+5])
            print('gather: ', labels_bank_i_l[2304:2304+5])

            local_text_key_bank_l = [torch.zeros_like(local_text_key_bank) for k in range(num_dev) ]
            print('!!!', args.local_rank, local_text_key_bank[0][0][0])
            dist.all_gather(local_text_key_bank_l, local_text_key_bank)
            local_text_key_bank_l = torch.cat(local_text_key_bank_l, dim=0)

            global_result, local_result, result = compute_topk(global_img_feat_bank, local_img_query_bank, local_img_value_bank, global_text_feat_bank, local_text_key_bank,
                                                            local_text_value_bank, length_bank, labels_bank_i, labels_bank_t, args, [1, 5, 10], True,False)
        '''
        global_result, local_result, result = compute_topk(global_img_feat_bank, local_img_query_bank, local_img_value_bank, global_text_feat_bank, local_text_key_bank,
                                                            local_text_value_bank, length_bank, labels_bank_i, labels_bank_t, args, [1, 5, 10], True,False)
        ac_top1_i2t, ac_top5_i2t, ac_top10_i2t, ac_top1_t2i, ac_top5_t2i, ac_top10_t2i = result
    
        return ac_top1_i2t, ac_top5_i2t, ac_top10_i2t, ac_top1_t2i, ac_top5_t2i , ac_top10_t2i, batch_time.avg



        


def main(args):
    # need to clear the pipeline
    # top1 & top10 need to be chosen in the same params ???
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    test_loader = data_config(args.image_dir, args.anno_dir, 64, 'test', 100, test_transform)
    unique_image = get_image_unique(args.image_dir, args.anno_dir, 64, 'test', 100, test_transform)

    
    
    

    # from /exp# dir get best_model.pth.tar
    for filename in glob.iglob(args.model_path + '/**/best_model.pth.tar', recursive=True):
        model_path = filename
    logging.info(model_path)
    network, _ = network_config(args, 'test', param=None, resume=False, model_path=model_path, param2=None)
    ac_top1_i2t, ac_top5_i2t, ac_top10_i2t, ac_top1_t2i, ac_top5_t2i , ac_top10_t2i, test_time = test(test_loader, network, args, unique_image)
    logging.info('top1_t2i: {:.3f}, top5_t2i: {:.3f}, top10_t2i: {:.3f}, top1_i2t: {:.3f}, top5_i2t: {:.3f}, top10_i2t: {:.3f}'.format(
            ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, ac_top1_i2t, ac_top5_i2t, ac_top10_i2t))



 

if __name__ == '__main__':
    args = config()
    main(args)
