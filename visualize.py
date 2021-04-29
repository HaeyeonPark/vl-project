import os
import sys
import time
import shutil
import logging
import gc
import torch
import torchvision.transforms as transforms
from utils.metric import AverageMeter, compute_topk
from vis_config import config
from config import data_config, network_config, get_image_unique
from textwrap import wrap
from PIL import Image
import numpy as np
import math
import re
import glob

import matplotlib.pyplot as plt

def visualize(data_loader, network, args, unique_image, image_path_list):
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
    caption_list = ()

    with torch.no_grad():
        end = time.time()
        for images, caption, labels in data_loader:
            captions = caption[0] + caption[1]
            caption_list += captions
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
        t2i_index, correct = compute_topk(global_img_feat_bank, local_img_query_bank, local_img_value_bank, global_text_feat_bank, local_text_key_bank,
                                                        local_text_value_bank, length_bank, labels_bank_i, labels_bank_t, args, [1, 5, 10], True, return_index=True)
        # save figure
        n = t2i_index.size(0)
        for i in range(0, n, 20):
            text_query = caption_list[i]
            fig = plt.figure()
            for k in range(10): 
                img_path = image_path_list[t2i_index[i][k]]
                ifcorrect = correct[i][k].item()

                fig.add_subplot(1,10,k+1)

                middle_path = "CUHK-PEDES/imgs"
                if middle_path not in img_path:
                    img_path = os.path.join(args.image_dir, middle_path, img_path)
                else:
                    img_path = os.path.join(args.image_root, img_path)
                im = Image.open(img_path)
                im=im.resize((150,400))
                im = np.asarray(im)
                #im = plt.imread(img_path)
                plt.imshow(im)
                plt.axis('off')
                plt.title(str(ifcorrect))
                #ax = plt.subplot(1,10,i+1)
                #ax.axis('off')
                #ax.set_title(str(ifcorrect))

                
            fig.suptitle("\n".join(wrap(text_query, 60)))
            fig.tight_layout()
            demo_path = os.path.join(args.model_path, 'demo')
            if not os.path.exists(demo_path):
                os.makedirs(demo_path)
            plt.savefig(os.path.join(demo_path,str(i)+'.png'))
            plt.close()


    
        return #ac_top1_i2t, ac_top5_i2t, ac_top10_i2t, ac_top1_t2i, ac_top5_t2i , ac_top10_t2i, batch_time.avg



def main(args):
    # need to clear the pipeline
    # top1 & top10 need to be chosen in the same params ???
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    test_loader, image_path_list = data_config(args.image_dir, args.anno_dir, 64, 'test', 100, test_transform, vis=True)
    unique_image = get_image_unique(args.image_dir, args.anno_dir, 64, 'test', 100, test_transform)

    
    
    

    # from /exp# dir get best_model.pth.tar
    for filename in glob.iglob(args.model_path + '/**/best_model.pth.tar', recursive=True):
        model_path = filename
    logging.info(model_path)
    network, _ = network_config(args, 'test', param=None, resume=False, model_path=model_path, param2=None)
    #ac_top1_i2t, ac_top5_i2t, ac_top10_i2t, ac_top1_t2i, ac_top5_t2i , ac_top10_t2i, test_time = test(test_loader, network, args, unique_image, image_path_list)
    visualize(test_loader, network, args, unique_image, image_path_list)
    #logging.info('top1_t2i: {:.3f}, top5_t2i: {:.3f}, top10_t2i: {:.3f}, top1_i2t: {:.3f}, top5_i2t: {:.3f}, top10_i2t: {:.3f}'.format(
    #       ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, ac_top1_i2t, ac_top5_i2t, ac_top10_i2t))




if __name__ == '__main__':
    args = config()
    main(args)
