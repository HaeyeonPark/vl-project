from utils.metric import compute_topk_combine
import os
import sys
import time
import shutil
import logging
import gc
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from utils.metric import AverageMeter, compute_topk, Loss
from test_config import config
from config import data_config, network_config, get_image_unique
import numpy as np
import math
import re
import glob

def test(data_loader, network, args, unique_image):
    batch_time = AverageMeter()

    # switch to evaluate mode
    network.eval()
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

            tokens, segments, input_masks, caption_length = network.module.language_model.pre_process(captions)
            sep_tokens, sep_segments, sep_input_masks, sep_caption_length = network.module.language_model.pre_process(sep_captions)

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

            global_img_feat, global_text_feat, local_img_query, local_img_value, local_text_key, local_text_value = network(images, tokens, segments, input_masks, sep_tokens, sep_segments, sep_input_masks, n_sep, p2, p3,  stage='train')

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

        #global_result, local_result, result = compute_topk(global_img_feat_bank[unique_image], local_img_query_bank[unique_image], local_img_value_bank[unique_image], global_text_feat_bank, local_text_key_bank,
        #                                                local_text_value_bank, length_bank, labels_bank[unique_image], labels_bank, args, [1, 5, 10], True)
        global_result, local_result, result = compute_topk(global_img_feat_bank, local_img_query_bank, local_img_value_bank, global_text_feat_bank, local_text_key_bank,
                                                        local_text_value_bank, length_bank, labels_bank_i, labels_bank_t, args, [1, 5, 10], True)

        ac_top1_i2t, ac_top5_i2t, ac_top10_i2t, ac_top1_t2i, ac_top5_t2i, ac_top10_t2i = result
    
        return ac_top1_i2t, ac_top5_i2t, ac_top10_i2t, ac_top1_t2i, ac_top5_t2i , ac_top10_t2i, batch_time.avg

def combine_test(data_loader, network, args, unique_image, loss):
    batch_time = AverageMeter()

    # switch to evaluate mode
    network.eval()
    max_size = 64 * len(data_loader)
    global_img_feat_bank = torch.zeros((max_size, args.feature_size)).cuda()
    global_text_feat_bank = torch.zeros((max_size, args.feature_size)).cuda()

    local_img_query_bank = torch.zeros((max_size, args.part2 + args.part3 + 1, args.feature_size)).cuda()
    local_img_value_bank = torch.zeros((max_size, args.part2 + args.part3 + 1, args.feature_size)).cuda()

    local_combined_text_feat_bank = torch.zeros((max_size, args.part2 + args.part3 + 1, args.feature_size)).cuda()
    local_text_key_bank = torch.zeros((max_size, 98 + 2 + 1, args.feature_size)).cuda()
    local_text_value_bank = torch.zeros((max_size, 98 + 2 + 1, args.feature_size)).cuda()

    labels_bank = torch.zeros(max_size).cuda()
    #length_bank = torch.zeros(max_size, dtype=torch.long).cuda()
    index = 0


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

            tokens, segments, input_masks, caption_length = network.module.language_model.pre_process(captions)
            sep_tokens, sep_segments, sep_input_masks, sep_caption_length = network.module.language_model.pre_process(sep_captions)

            tokens = tokens.cuda()
            segments = segments.cuda()
            input_masks = input_masks.cuda()
            caption_length = caption_length.cuda()

            sep_tokens = sep_tokens.cuda()
            sep_segments = sep_segments.cuda()
            sep_input_masks = sep_input_masks.cuda()
            
            images = images.cuda()
            labels = labels.cuda()
            interval = images.shape[0]

            p2 = [i for i in range(args.part2)]
            p3 = [i for i in range(args.part3)]

            global_img_feat, global_text_feat, local_img_query, local_img_value, local_text_key, local_text_value = network(images, tokens, segments, input_masks, sep_tokens, sep_segments, sep_input_masks, n_sep, p2, p3,  stage='train')
            weiTexts = loss.compute_weiTexts(local_img_query, local_text_key, local_text_value, caption_length, args)
            combineTexts = loss.compute_combineTexts(weiTexts)


            global_img_feat_bank[index: index + interval] = global_img_feat
            global_text_feat_bank[index: index + interval] = combineTexts[:,0,:]
            local_img_query_bank[index: index + interval, :, :] = local_img_query
            local_img_value_bank[index: index + interval, :, :] = local_img_value

            local_combined_text_feat_bank[index: index + interval,:, :] = combineTexts
            #local_text_key_bank[index: index + interval, :, :] = local_text_key
            #local_text_value_bank[index: index + interval, :, :] = local_text_value
            labels_bank[index:index + interval] = labels
            #labels_bank_t[index_t:index_t + interval_t] = torch.cat((labels, labels))
            #length_bank[index:index + interval] = caption_length
            batch_time.update(time.time() - end)
            end = time.time()
            index = index + interval

        global_img_feat_bank = global_img_feat_bank[:index]
        global_text_feat_bank = global_text_feat_bank[:index]
        local_img_query_bank = local_img_query_bank[:index]
        local_img_value_bank = local_img_value_bank[:index]
        local_combined_text_feat_bank = local_combined_text_feat_bank[:index]
        #local_text_key_bank = local_text_key_bank[:index]
        #local_text_value_bank = local_text_value_bank[:index]
        labels_bank = labels_bank[:index]
        #labels_bank_t = labels_bank[:index]
        #length_bank = length_bank[:index]
        unique_image = torch.tensor(unique_image) == 1

        #global_result, local_result, result = compute_topk(global_img_feat_bank[unique_image], local_img_query_bank[unique_image], local_img_value_bank[unique_image], global_text_feat_bank, local_text_key_bank,
        #                                                local_text_value_bank, length_bank, labels_bank[unique_image], labels_bank, args, [1, 5, 10], True)
        global_result, local_result, result = compute_topk_combine(global_img_feat_bank, local_img_query_bank, local_img_value_bank, global_text_feat_bank, 
                                                        local_combined_text_feat_bank, labels_bank, labels_bank, args, [1, 5, 10], True)

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

    compute_loss = Loss(args)
    nn.DataParallel(compute_loss).cuda()

    if args.test_type == 'combine':
        ac_top1_i2t, ac_top5_i2t, ac_top10_i2t, ac_top1_t2i, ac_top5_t2i , ac_top10_t2i, test_time = combine_test(test_loader, network, args, unique_image, compute_loss)
        logging.info('top1_t2i: {:.3f}, top5_t2i: {:.3f}, top10_t2i: {:.3f}, top1_i2t: {:.3f}, top5_i2t: {:.3f}, top10_i2t: {:.3f}'.format(
                ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, ac_top1_i2t, ac_top5_i2t, ac_top10_i2t))

   
    else: 
        ac_top1_i2t, ac_top5_i2t, ac_top10_i2t, ac_top1_t2i, ac_top5_t2i , ac_top10_t2i, test_time = test(test_loader, network, args, unique_image)
        logging.info('top1_t2i: {:.3f}, top5_t2i: {:.3f}, top10_t2i: {:.3f}, top1_i2t: {:.3f}, top5_i2t: {:.3f}, top10_i2t: {:.3f}'.format(
                ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, ac_top1_i2t, ac_top5_i2t, ac_top10_i2t))



    '''
    i2t_models = os.listdir(args.model_path)
    i2t_models.sort()
    model_list = []
    for i2t_model in i2t_models:
        if i2t_model.split('.')[0] != "model_best":
            model_list.append(int(i2t_model.split('.')[0]))
        model_list.sort()
    '''
   
    ''' for debug
    
    logging.info('Testing on dataset: {}'.format(args.anno_dir))
    network, _ = network_config(args, 'test')

    ac_top1_i2t, ac_top5_i2t, ac_top10_i2t, ac_top1_t2i, ac_top5_t2i , ac_top10_t2i, test_time = test(test_loader, network, args, unique_image)
    logging.info('top1_t2i: {:.3f}, top5_t2i: {:.3f}, top10_t2i: {:.3f}, top1_i2t: {:.3f}, top5_i2t: {:.3f}, top10_i2t: {:.3f}'.format(
            ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, ac_top1_i2t, ac_top5_i2t, ac_top10_i2t))
    
    '''
    '''
    ac_i2t_top1_best = 0.0
    ac_i2t_top10_best = 0.0
    ac_t2i_top1_best = 0.0
    ac_t2i_top10_best = 0.0
    ac_t2i_top5_best = 0.0
    ac_i2t_top5_best = 0.0
    for i2t_model in model_list:
        model_file = os.path.join(args.model_path, str(i2t_model) + '.pth.tar')
        if os.path.isdir(model_file):
            continue
        epoch = i2t_model
        if int(epoch) < args.epoch_start:
            continue
        network, _ = network_config(args, 'test', None, True)

        ac_top1_i2t, ac_top5_i2t, ac_top10_i2t, ac_top1_t2i, ac_top5_t2i , ac_top10_t2i, test_time = test(test_loader, network, args, unique_image)
        if ac_top1_t2i > ac_t2i_top1_best:
            ac_i2t_top1_best = ac_top1_i2t
            ac_i2t_top5_best = ac_top5_i2t
            ac_i2t_top10_best = ac_top10_i2t

            ac_t2i_top1_best = ac_top1_t2i
            ac_t2i_top5_best = ac_top5_t2i
            ac_t2i_top10_best = ac_top10_t2i
            dst_best = os.path.join(args.checkpoint_dir, 'model_best', str(epoch)) + '.pth.tar'
        

        logging.info('epoch:{}'.format(epoch))
        logging.info('top1_t2i: {:.3f}, top5_t2i: {:.3f}, top10_t2i: {:.3f}, top1_i2t: {:.3f}, top5_i2t: {:.3f}, top10_i2t: {:.3f}'.format(
            ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, ac_top1_i2t, ac_top5_i2t, ac_top10_i2t))
    logging.info('t2i_top1_best: {:.3f}, t2i_top5_best: {:.3f}, t2i_top10_best: {:.3f}, i2t_top1_best: {:.3f}, i2t_top5_best: {:.3f}, i2t_top10_best: {:.3f}'.format(
        ac_t2i_top1_best, ac_t2i_top5_best, ac_t2i_top10_best, ac_i2t_top1_best, ac_i2t_top5_best, ac_i2t_top10_best))
    logging.info(args.model_path)
    logging.info(args.log_dir)
    '''

if __name__ == '__main__':
    args = config()
    main(args)