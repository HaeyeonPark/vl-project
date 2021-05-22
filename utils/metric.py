import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import pickle

from torch.autograd import Variable

from math import exp

logger = logging.getLogger()                                                                                                                                                                            
logger.setLevel(logging.INFO)


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    #norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    #X = torch.div(X, norm)
    return F.normalize(X, dim=dim, p=2)


def compute_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def pairwise_distance(A, B):
    """
    Compute distance between points in A and points in B
    :param A:  (m,n) -m points, each of n dimension. Every row vector is a point, denoted as A(i).
    :param B:  (k,n) -k points, each of n dimension. Every row vector is a point, denoted as B(j).
    :return:  Matrix with (m, k). And the ele in (i,j) is the distance between A(i) and B(j)
    """
    A_square = torch.sum(A * A, dim=1, keepdim=True)
    B_square = torch.sum(B * B, dim=1, keepdim=True)

    distance = A_square + B_square.t() - 2 * torch.matmul(A, B.t())

    return distance


def func_attention_MxN(local_img_query, txt_i_key_expand, txt_i_value_expand, opt, eps=1e-8):
    """
    query: (batch, queryL, d)
    context: (batch, sourceL, d)
    opt: parameters
    """
    # 16, 6, 25
    batch_size, queryL, sourceL = txt_i_key_expand.size(
        0), local_img_query.size(1), txt_i_key_expand.size(1)
    local_img_query_norm = l2norm(local_img_query, dim=-1)
    txt_i_key_expand_norm = l2norm(txt_i_key_expand, dim=-1)

    # Step 1: preassign attention
    # --> (batch, d, queryL)
    local_img_queryT = torch.transpose(local_img_query_norm, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    attn = torch.bmm(txt_i_key_expand_norm, local_img_queryT)
    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax(dim=1)(attn * opt.lambda_softmax)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # print('attn: ', attn)

    # Step 2: identify irrelevant fragments
    # Learning an indicator function H, one for relevant, zero for irrelevant
    if opt.focal_type == 'equal':
        funcH = focal_equal(attn, batch_size, queryL, sourceL)
    elif opt.focal_type == 'prob':
        funcH = focal_prob(attn, batch_size, queryL, sourceL)
    else:
        funcH = None
    

    # Step 3: reassign attention
    if funcH is not None:
        tmp_attn = funcH * attn
        attn_sum = torch.sum(tmp_attn, dim=-1, keepdim=True)
        attn = tmp_attn / attn_sum

    # --> (batch, d, sourceL)
    txt_i_valueT = torch.transpose(txt_i_value_expand, 1, 2)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(txt_i_valueT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    #return weightedContext, attn
    return weightedContext

def focal_equal(attn, batch_size, queryL, sourceL):
    """
    consider the confidence g(x) for each fragment as equal
    sigma_{j} (xi - xj) = sigma_{j} xi - sigma_{j} xj
    attn: (batch, queryL, sourceL)
    """
    funcF = attn * sourceL - torch.sum(attn, dim=-1, keepdim=True)
    fattn = torch.where(funcF > 0, torch.ones_like(attn),
                        torch.zeros_like(attn))
    return fattn


def focal_prob(attn, batch_size, queryL, sourceL):
    """
    consider the confidence g(x) for each fragment as the sqrt
    of their similarity probability to the query fragment
    sigma_{j} (xi - xj)gj = sigma_{j} xi*gj - sigma_{j} xj*gj
    attn: (batch, queryL, sourceL)
    """

    # -> (batch, queryL, sourceL, 1)
    xi = attn.unsqueeze(-1).contiguous()
    # -> (batch, queryL, 1, sourceL)
    xj = attn.unsqueeze(2).contiguous()
    # -> (batch, queryL, 1, sourceL)
    xj_confi = torch.sqrt(xj)

    xi = xi.view(batch_size * queryL, sourceL, 1)
    xj = xj.view(batch_size * queryL, 1, sourceL)
    xj_confi = xj_confi.view(batch_size * queryL, 1, sourceL)

    # -> (batch*queryL, sourceL, sourceL)
    term1 = torch.bmm(xi, xj_confi)
    term2 = xj * xj_confi
    funcF = torch.sum(term1 - term2, dim=-1)  # -> (batch*queryL, sourceL)
    funcF = funcF.view(batch_size, queryL, sourceL)

    fattn = torch.where(funcF > 0, torch.ones_like(attn),
                        torch.zeros_like(attn))
    return fattn


def constraints(features, labels):
    labels = torch.reshape(labels, (labels.shape[0],1))
    con_loss = AverageMeter()
    index_dict = {k.item() for k in labels}
    for index in index_dict:
        labels_mask = (labels == index)
        feas = torch.masked_select(features, labels_mask)
        feas = feas.view(-1, features.shape[1])
        distance = pairwise_distance(feas, feas)
        #torch.sqrt_(distance)
        num = feas.shape[0] * (feas.shape[0] - 1)
        loss = torch.sum(distance) / num
        con_loss.update(loss, n = num / 2)
    return con_loss.avg


def constraints_loss(data_loader, network, args):
    network.eval()
    max_size = args.batch_size * len(data_loader)
    images_bank = torch.zeros((max_size, args.feature_size)).cuda()
    text_bank = torch.zeros((max_size,args.feature_size)).cuda()
    labels_bank = torch.zeros(max_size).cuda()
    index = 0
    con_images = 0.0
    con_text = 0.0
    with torch.no_grad():
        for images, captions, labels, captions_length in data_loader:
            images = images.cuda()
            captions = captions.cuda()
            interval = images.shape[0]
            image_embeddings, text_embeddings = network(images, captions, captions_length)
            images_bank[index: index + interval] = image_embeddings
            text_bank[index: index + interval] = text_embeddings
            labels_bank[index: index + interval] = labels
            index = index + interval
        images_bank = images_bank[:index]
        text_bank = text_bank[:index]
        labels_bank = labels_bank[:index]
    
    if args.constraints_text:
        con_text = constraints(text_bank, labels_bank)
    if args.constraints_images:
        con_images = constraints(images_bank, labels_bank)

    return con_images, con_text
   

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.args = args
        self.CMPM = args.CMPM
        self.CMPC = args.CMPC
        self.COMBINE = args.COMBINE
        self.CONT = args.CONT
        self.PART_I2T = args.PART_I2T
        self.PART_CBT2I = args.PART_CBT2I
        self.epsilon = args.epsilon
        self.num_classes = args.num_classes
        self.cb_layer = nn.Linear(args.feature_size*2, args.feature_size)
        self.W = Parameter(torch.randn(args.feature_size, args.num_classes))
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.W.data, gain=1)
        nn.init.xavier_uniform_(self.cb_layer.weight, gain=1)

    @staticmethod
    def compute_i2t_sim(local_img_query, local_img_value, local_text_key, local_text_value, text_length, args):
        """
        Compute weighted text embeddings
        :param image_embeddings: Tensor with dtype torch.float32, [n_img, n_region, d]
        :param text_embeddings: Tensor with dtype torch.float32, [n_txt, n_word, d]
        :param text_length: list, contain length of each sentence, [batch_size]
        :param labels: Tensor with dtype torch.int32, [batch_size]
        :return: i2t_similarities: Tensor, [n_img, n_txt]
                 t2i_similarities: Tensor, [n_img, n_txt]
        """
        n_img = local_img_query.shape[0]
        n_txt = local_text_key.shape[0]
        t2i_similarities = []
        i2t_similarities = []
        #atten_final_result = {}
        for i in range(n_txt):
            # Get the i-th text description
            n_word = text_length[i]
            # why? local text contains global then n_word+1 isn't it??
            txt_i_key = local_text_key[i, :n_word, :].unsqueeze(0).contiguous()
            txt_i_value = local_text_value[i, :n_word, :].unsqueeze(0).contiguous()
            # -> (n_img, n_word, d)
            txt_i_key_expand = txt_i_key.repeat(n_img, 1, 1)
            txt_i_value_expand = txt_i_value.repeat(n_img, 1, 1)

            # -> (n_img, n_region, d)
            #weiText, atten_text = func_attention_MxN(local_img_query, txt_i_key_expand, txt_i_value_expand, args)
            weiText = func_attention_MxN(local_img_query, txt_i_key_expand, txt_i_value_expand, args)
            #atten_final_result[i] = atten_text[i, :, :]
            # image_embeddings = l2norm(image_embeddings, dim=2)
            weiText = l2norm(weiText, dim=2)
            i2t_sim = compute_similarity(local_img_value, weiText, dim=2)
            i2t_sim = i2t_sim.mean(dim=1, keepdim=True)
            i2t_similarities.append(i2t_sim)

            '''
            # -> (n_img, n_word, d)
            #weiImage, atten_image = func_attention_MxN(txt_i_key_expand, local_img_query, local_img_value, args)
            weiImage = func_attention_MxN(txt_i_key_expand, local_img_query, local_img_value, args)
            # txt_i_expand = l2norm(txt_i_expand, dim=2)
            weiImage = l2norm(weiImage, dim=2)
            t2i_sim = compute_similarity(txt_i_value_expand, weiImage, dim=2)
            t2i_sim = t2i_sim.mean(dim=1, keepdim=True)
            # images ~ text similarity 16*1 
            t2i_similarities.append(t2i_sim)
            '''
            del txt_i_key_expand
            del txt_i_value_expand

        # img * txt * part * dim 
        i2t_similarities = torch.cat(i2t_similarities, 1)
        #t2i_similarities = torch.cat(t2i_similarities, 1)

        return i2t_similarities #, t2i_similarities


    @staticmethod
    def compute_weiTexts(local_img_query, local_text_key, local_text_value, text_length, args):
        """
        Compute weighted text embeddings
        :param image_embeddings: Tensor with dtype torch.float32, [n_img, n_region, d]
        :param text_embeddings: Tensor with dtype torch.float32, [n_txt, n_word, d]
        :param text_length: list, contain length of each sentence, [batch_size]
        :param labels: Tensor with dtype torch.int32, [batch_size]
        :return: i2t_similarities: Tensor, [n_img, n_txt]
                 t2i_similarities: Tensor, [n_img, n_txt]
        """
        n_img = local_img_query.shape[0]
        n_txt = local_text_key.shape[0]
        t2i_similarities = []
        i2t_similarities = []
        #atten_final_result = {}
        for i in range(n_txt):
            # Get the i-th text description
            n_word = text_length[i]
            # why? local text contains global then n_word+1 isn't it??
            txt_i_key = local_text_key[i, :n_word, :].unsqueeze(0).contiguous()
            txt_i_value = local_text_value[i, :n_word, :].unsqueeze(0).contiguous()
            # -> (n_img, n_word, d)
            #txt_i_key_expand = txt_i_key.repeat(n_img, 1, 1)
            #txt_i_value_expand = txt_i_value.repeat(n_img, 1, 1)
            txt_i_key_expand = txt_i_key.expand(n_img, n_word, args.feature_size)
            txt_i_value_expand = txt_i_value.expand(n_img, n_word, args.feature_size)

            # -> (n_img, n_region, d)
            #weiText, atten_text = func_attention_MxN(local_img_query, txt_i_key_expand, txt_i_value_expand, args)
            weiText = func_attention_MxN(local_img_query, txt_i_key_expand, txt_i_value_expand, args)
            #atten_final_result[i] = atten_text[i, :, :]
            # image_embeddings = l2norm(image_embeddings, dim=2)
            weiText = l2norm(weiText, dim=2)
            i2t_similarities.append(weiText.unsqueeze(1))
            '''
            i2t_sim = compute_similarity(local_img_value, weiText, dim=2)
            i2t_sim = i2t_sim.mean(dim=1, keepdim=True)
            i2t_similarities.append(i2t_sim)

            # -> (n_img, n_word, d)
            #weiImage, atten_image = func_attention_MxN(txt_i_key_expand, local_img_query, local_img_value, args)
            weiImage = func_attention_MxN(txt_i_key_expand, local_img_query, local_img_value, args)
            # txt_i_expand = l2norm(txt_i_expand, dim=2)
            weiImage = l2norm(weiImage, dim=2)
            t2i_sim = compute_similarity(txt_i_value_expand, weiImage, dim=2)
            t2i_sim = t2i_sim.mean(dim=1, keepdim=True)
            # images ~ text similarity 16*1 
            t2i_similarities.append(t2i_sim)
            '''
            del txt_i_key_expand
            del txt_i_value_expand

        # img * txt * part * dim 
        i2t_similarities = torch.cat(i2t_similarities, 1)
        #t2i_similarities = torch.cat(t2i_similarities, 1)

        return i2t_similarities #, t2i_similarities

    def contrastive_loss(self, local_img_value, weiTexts, labels):
        # i2t
        n_i, n_t, n_p, dim = weiTexts.size()
        local_img_expand = local_img_value.expand(n_t, n_i, n_p, dim)

        weiTexts_T = weiTexts.transpose(0,1)
        i2t_sim = compute_similarity(local_img_expand, weiTexts_T, dim=-1) # 32*16*6
        i2t_sim = i2t_sim.mean(dim=-1) # 32*16
        i2t_pred = F.softmax(i2t_sim * self.args.lambda_softmax, dim=0)

        labels_reshape = labels.unsqueeze(1)
        labels_dist = labels_reshape - labels_reshape.t()
        labels_mask = (labels_dist == 0)
        labels_mask = torch.cat((labels_mask, labels_mask), dim=0)
        labels = labels_mask.float() / labels_mask.float().norm(p=1, dim=0)

        cont_loss = i2t_pred * (torch.log(i2t_pred) - torch.log(labels + self.args.epsilon))
        cont_loss = torch.sum(cont_loss, dim=0).mean()

        pos_avg_sim = torch.mean(torch.masked_select(i2t_sim, labels_mask))
        neg_avg_sim = torch.mean(torch.masked_select(i2t_sim, labels_mask == 0))

        return cont_loss, pos_avg_sim, neg_avg_sim


    def compute_cmpc_loss(self, image_embeddings, text_embeddings, labels):
        """
        Cross-Modal Projection Classfication loss(CMPC)
        :param image_embeddings: Tensor with dtype torch.float32
        :param text_embeddings: Tensor with dtype torch.float32
        :param labels: Tensor with dtype torch.int32
        :return:
        """
        criterion = nn.CrossEntropyLoss(reduction='mean')
        self.W_norm = F.normalize(self.W, p=2, dim=0)

        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

        # two captions per image
        image_embeddings = torch.cat((image_embeddings,image_embeddings), dim=0)
        image_norm = torch.cat((image_norm, image_norm), dim=0)
        labels = torch.cat((labels, labels), dim=0)

        image_proj_text = torch.sum(image_embeddings * text_norm, dim=1, keepdim=True) * text_norm
        text_proj_image = torch.sum(text_embeddings * image_norm, dim=1, keepdim=True) * image_norm

        image_logits = torch.matmul(image_proj_text, self.W_norm)
        text_logits = torch.matmul(text_proj_image, self.W_norm)
        

        # image_logits = torch.matmul(image_embeddings, self.W_norm)
        # text_logits = torch.matmul(text_embeddings, self.W_norm)

        '''
        ipt_loss = criterion(input=image_logits, target=labels)
        tpi_loss = criterion(input=text_logits, target=labels)
        cmpc_loss = ipt_loss + tpi_loss
        '''
        cmpc_loss = criterion(image_logits, labels) + criterion(text_logits, labels)

        # classification accuracy for observation
        image_pred = torch.argmax(image_logits, dim=1)
        text_pred = torch.argmax(text_logits, dim=1)

        image_precision = torch.mean((image_pred == labels).float())
        text_precision = torch.mean((text_pred == labels).float())

        return cmpc_loss, image_precision, text_precision

    def compute_cmpm_loss(self, image_embeddings, text_embeddings, labels):
        """
        Cross-Modal Projection Matching Loss(CMPM)
        :param image_embeddings: Tensor with dtype torch.float32
        :param text_embeddings: Tensor with dtype torch.float32
        :param labels: Tensor with dtype torch.int32
        :return:
            i2t_loss: cmpm loss for image projected to text
            t2i_loss: cmpm loss for text projected to image
            pos_avg_sim: average cosine-similarity for positive pairs
            neg_avg_sim: averate cosine-similarity for negative pairs
        """

        batch_size = image_embeddings.shape[0]

        # print("batch size: " + str(batch_size))

        labels_reshape = torch.reshape(labels, (batch_size, 1))
        labels_reshape_text = torch.cat((labels_reshape, labels_reshape), 0)
        labels_dist = labels_reshape - labels_reshape_text.t()
        labels_mask = (labels_dist == 0)

        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
        image_proj_text = torch.matmul(image_embeddings, text_norm.t())
        text_proj_image = torch.matmul(text_embeddings, image_norm.t())

        # normalize the true matching distribution
        #labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)
        labels_mask_t = labels_mask.t()
        labels_mask_norm_i2t = (labels_mask_t.float() / labels_mask_t.float().norm(dim=0,p=1)).t()
        labels_mask_norm_t2i = (labels_mask.float() / labels_mask.float().norm(dim=0,p=1)).t()

        i2t_pred = F.softmax(image_proj_text, dim=1)
        # i2t_loss = i2t_pred * torch.log((i2t_pred + self.epsilon)/ (labels_mask_norm + self.epsilon))
        i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm_i2t + self.epsilon))

        t2i_pred = F.softmax(text_proj_image, dim=1)
        # t2i_loss = t2i_pred * torch.log((t2i_pred + self.epsilon)/ (labels_mask_norm + self.epsilon))
        t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm_t2i + self.epsilon))

        cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
        # its for showing similarity between positive pairs and negative pairs
        sim_cos = torch.matmul(image_norm, text_norm.t())

        pos_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask))
        neg_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask == 0))

        return cmpm_loss, pos_avg_sim, neg_avg_sim

    def compute_combineTexts(self, wei_text):
        # compute part level combined texts
        # 1> use one fc layer to combine two txt vectors
        concat = []
        n = wei_text.size(0)
        for i in range(n):
            concat.append(torch.cat((wei_text[i,i,:,:], wei_text[i,i+n,:,:]), dim=1))
        combineTexts = torch.stack(concat, dim=0)
        combineTexts = self.cb_layer(combineTexts)
        #combineTexts = l2norm(combineTexts, dim=2)
        return combineTexts

    def compute_combine_loss(self, combineTexts, local_img_value, labels, lambda_cb, epsilon):
        #to align combined texts to image
        '''
        # matrix ver + not add partwise
        labels_reshape = labels.unsqueeze(1)
        labels_dist = labels_reshape - labels_reshape.t()
        labels_mask = (labels_dist == 0)
        labels_mask = labels_mask.repeat(combineTexts.size(1),1,1).float()

        local_img_value_norm = l2norm(local_img_value, dim=-1)
        local_img_value_T = local_img_value_norm.transpose(0,1)

        combineTexts_norm = l2norm(combineTexts, dim=-1)
        combinedTexts_T = combineTexts_norm.permute(1,2,0)

        sim = torch.bmm(local_img_value_T, combinedTexts_T)
        pred = F.softmax(sim * lambda_cb, dim=2)

        combine_loss = pred * (torch.log(pred) - torch.log(labels_mask + epsilon))
        combine_loss = torch.sum(combine_loss, dim=2).mean()
        '''
        n = local_img_value.size(0)
        i2t_sim=[]

        labels_reshape = labels.unsqueeze(1)
        labels_dist = labels_reshape - labels_reshape.t()
        labels_mask = (labels_dist==0)
        labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)

        for i in range(n):
            # for each image  + all combined texts
            img_i_expand = local_img_value[i,:,:].repeat(n,1,1)
            sim = compute_similarity(img_i_expand, combineTexts, dim=2) 
            sim = sim.mean(dim=1)
            i2t_sim.append(sim)
        i2t_sim = torch.stack(i2t_sim, dim=0) # n * 16(txt) 
        i2t_pred = F.softmax(i2t_sim * self.args.lambda_softmax, dim=1)
        combine_loss = i2t_pred * (torch.log(i2t_pred) - torch.log(labels_mask_norm + self.epsilon))
        combine_loss = torch.mean(torch.sum(combine_loss, dim=1))

        cb_pos_avg_sim = torch.mean(torch.masked_select(i2t_sim, labels_mask))
        cb_neg_avg_sim = torch.mean(torch.masked_select(i2t_sim, labels_mask==0))
        return combine_loss, cb_pos_avg_sim, cb_neg_avg_sim

    def compute_part_loss(self, weiTexts, combineTexts, local_img_value, PART_I2T, PART_CBT2I):
        # i2t
        part_result = {}
        part_loss = 0
        if PART_I2T:
            n_i, n_t, n_p, dim = weiTexts.size()
            local_img_expand = local_img_value.expand(n_t, n_i, n_p, dim)
            local_img_expand = local_img_expand.transpose(0,1)

            combineTexts_expand = combineTexts.expand(n_t, n_i, n_p, dim)
            combineTexts_expand = combineTexts_expand.transpose(0,1)

            i2t_sim = compute_similarity(local_img_expand, weiTexts, dim=3)
            i2t_pred = F.softmax(i2t_sim * self.args.lambda_softmax, dim=1)

            i2cbt_sim = compute_similarity(combineTexts_expand, weiTexts, dim=3)
            i2cbt_pred = F.softmax(i2cbt_sim * self.args.lambda_softmax, dim=1)
            part_i2t_loss = i2t_pred * (torch.log(i2t_pred) - torch.log(i2cbt_pred + self.epsilon))
            part_i2t_loss = torch.sum(part_i2t_loss, dim=1)

            each_part_i2t_loss = torch.mean(part_i2t_loss, dim=0) # each part
            each_part_i2t_loss = each_part_i2t_loss.detach().to('cpu').numpy()
            part_result['each_part_i2t_loss'] = each_part_i2t_loss

            part_i2t_loss = torch.mean(part_i2t_loss)
            part_loss +=part_i2t_loss

        # cbt2i
        if PART_CBT2I:
            local_img_expand_t2i = local_img_value.expand(n_i, n_i, n_p, dim)
            combineTexts_expand_set = combineTexts.expand(n_i, n_i, n_p, dim)
            combineTexts_expand_t2i = combineTexts_expand_set.transpose(0,1)

            cbt2i_sim = compute_similarity(local_img_expand_t2i, combineTexts_expand_t2i, dim=3)
            cbt2i_pred = F.softmax(cbt2i_sim* self.args.lambda_softmax, dim=1)

            cbt2cbt_sim = compute_similarity(combineTexts_expand_t2i, combineTexts_expand_set, dim=3)
            cbt2cbt_pred = F.softmax(cbt2cbt_sim * self.args.lambda_softmax, dim=1)

            part_t2i_loss = cbt2i_pred * (torch.log(cbt2i_pred) - torch.log(cbt2cbt_pred + self.epsilon))
            part_t2i_loss = torch.sum(part_t2i_loss, dim=1)

            each_part_t2i_loss = torch.mean(part_t2i_loss, dim=0) # each part
            each_part_t2i_loss = each_part_t2i_loss.detach().to('cpu').numpy()
            part_result['each_part_t2i_loss'] = each_part_t2i_loss
            
            part_t2i_loss = torch.mean(part_t2i_loss)
            part_loss += part_t2i_loss
            
        part_result['part_loss'] = part_loss 
        return part_result

    def forward(self, total_epoch, epoch, global_img_feat, global_text_feat, local_img_query, local_img_value, local_text_key, local_text_value, text_length,
                labels):
        loss = 0
        result_dict = {}
        if self.CMPM:
            cmpm_loss, pos_avg_sim, neg_avg_sim = self.compute_cmpm_loss(global_img_feat, global_text_feat,
                                                                         labels)
            loss += cmpm_loss
            result_dict['cmpm_loss'] = cmpm_loss.item()
            result_dict['pos_avg_sim'] = pos_avg_sim
            result_dict['neg_avg_sim'] = neg_avg_sim
        if self.CMPC:
            cmpc_loss, image_precision, text_precision = self.compute_cmpc_loss(global_img_feat,
                                                                                global_text_feat, labels)
            loss += cmpc_loss
            result_dict['cmpc_loss'] = cmpc_loss.item()
            result_dict['image_precision'] = image_precision
            result_dict['text_precision'] = text_precision

        if self.COMBINE or self.PART_I2T or self.PART_CBT2I or self.CONT:
            weiTexts = self.compute_weiTexts(local_img_query, local_text_key, local_text_value, text_length, self.args)
            if self.COMBINE or self.PART_I2T or self.PART_CBT2I:
                combineTexts = self.compute_combineTexts(weiTexts)

        if self.COMBINE:
            # image based attended weighted vectors 16 * 32 * 6 * 768
            combine_loss, cb_pos_avg_sim, cb_neg_avg_sim = self.compute_combine_loss(combineTexts, local_img_value, labels, self.args.lambda_softmax, self.args.epsilon)
            combine_loss = combine_loss * self.args.lambda_combine
            loss += combine_loss
            result_dict['combine_loss'] = combine_loss.item()
            result_dict['cb_pos_avg_sim'] = cb_pos_avg_sim
            result_dict['cb_neg_avg_sim'] = cb_neg_avg_sim

        if self.PART_I2T or self.PART_CBT2I:
            # i2t + t2i
            part_result = self.compute_part_loss(weiTexts, combineTexts, local_img_value, self.PART_I2T, self.PART_CBT2I)

            #part_loss, each_part_i2t, each_part_t2i = self.compute_part_loss(weiTexts, combineTexts, local_img_value, self.PART_I2T, self.PART_CBT2I)
            #part_loss = part_loss * self.args.lambda_cont * min(1, exp(beta)+ beta -1)
            #result_dict['part_loss'] = part_loss.item()

            beta = epoch / total_epoch
            part_result['part_loss'] *= self.args.lambda_combine * min(1, exp(beta) + beta -1)
            loss += part_result['part_loss']
            part_result['part_loss'] = part_result['part_loss'].item()
            result_dict.update(part_result)

        if self.CONT:
            cont_loss, local_pos_avg_sim, local_neg_avg_sim = self.contrastive_loss(local_img_value, weiTexts, labels)
            cont_loss = cont_loss * self.args.lambda_cont
            loss += cont_loss
            result_dict['cont_loss'] = cont_loss.item()
            result_dict['local_pos_avg_sim'] = local_pos_avg_sim
            result_dict['local_neg_avg_sim'] = local_neg_avg_sim

        return loss, result_dict


class AverageMeter(object):
    """
    Computes and stores the averate and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py #L247-262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += n * val
        self.count += n
        self.avg = self.sum / self.count

def compute_topk_combine(query_global, query, value_bank, gallery_global, gallery_feat,
                       gallery_length, target_query, target_gallery, args, k_list=[1, 5, 20], reverse=False):
    global_result = []
    local_result = []
    result = []
    sim_cosine = []

    query_global = F.normalize(query_global, p=2, dim=1)
    gallery_global = F.normalize(gallery_global, p=2, dim=1)

    sim_cosine_global = torch.matmul(query_global, gallery_global.t())

    
    # compute_i2t_sim : weighted text구해서 similarity계산까지 함
    # combine test라면 weighted text구하고 합쳐진 상태이므로 compute similarity함수이용해서 해주면 됨 여기수정 !!
    # 원래도 local text에 맨앞이 global 이었당. 
    # 그리고 gallery_key, gallery_value parameter로 없애도 된당,, 
    # compute_topk_combine을 만들어야 할듯 

    for i in range(0, query.shape[0], 200):
        i2t_sim = Loss.compute_i2t_sim(query[i:i + 200], value_bank[i:i + 200], gallery_key, gallery_value, gallery_length, args)
        sim_cosine.append(i2t_sim)

    sim_cosine = torch.cat(sim_cosine, dim=0)
    
    sim_cosine_all = sim_cosine_global + sim_cosine
    reid_sim = None
    if(args.reranking):
        reid_sim = torch.matmul(query_global, query_global.t())

    global_result.extend(topk(sim_cosine_global, target_gallery, target_query, k=k_list))
    if reverse:
        global_result.extend(topk(sim_cosine_global, target_query, target_gallery, k=k_list, dim=0, print_index=False))

    local_result.extend(topk(sim_cosine, target_gallery, target_query, k=k_list))
    if reverse:
        local_result.extend(topk(sim_cosine, target_query, target_gallery, k=k_list, dim=0, print_index=False))

    # i2t
    result.extend(topk(sim_cosine_all, target_gallery, target_query, k=k_list, reid_sim=reid_sim))
    # t2i
    if reverse:
        result.extend(topk(sim_cosine_all, target_query, target_gallery, k=k_list, dim=0, print_index=False, reid_sim=reid_sim))
    return global_result, local_result, result


def compute_topk(query_global, query, value_bank, gallery_global, gallery_key, gallery_value,
                       gallery_length, target_query, target_gallery, args, k_list=[1, 5, 20], reverse=False, return_index=False):
    global_result = []
    local_result = []
    result = []
    sim_cosine = []

    query_global = F.normalize(query_global, p=2, dim=1)
    gallery_global = F.normalize(gallery_global, p=2, dim=1)

    sim_cosine_global = torch.matmul(query_global, gallery_global.t())


    for i in range(0, query.shape[0], 200):
        i2t_sim = Loss.compute_i2t_sim(query[i:i + 200], value_bank[i:i + 200], gallery_key, gallery_value, gallery_length, args)
        sim_cosine.append(i2t_sim)

    sim_cosine = torch.cat(sim_cosine, dim=0)
    
    sim_cosine_all = sim_cosine_global + sim_cosine
    reid_sim = None
    if(args.reranking):
        reid_sim = torch.matmul(query_global, query_global.t())

    global_result.extend(topk(sim_cosine_global, target_gallery, target_query, k=k_list))
    if reverse:
        global_result.extend(topk(sim_cosine_global, target_query, target_gallery, k=k_list, dim=0, print_index=False))

    local_result.extend(topk(sim_cosine, target_gallery, target_query, k=k_list))
    if reverse:
        local_result.extend(topk(sim_cosine, target_query, target_gallery, k=k_list, dim=0, print_index=False))

    if return_index==False:
        # i2t
        result.extend(topk(sim_cosine_all, target_gallery, target_query, k=k_list, reid_sim=reid_sim))
        # t2i
        if reverse:
            result.extend(topk(sim_cosine_all, target_query, target_gallery, k=k_list, dim=0, print_index=False, reid_sim=reid_sim))
        return global_result, local_result, result

    elif return_index == True:
        pred_index, correct = topk(sim_cosine_all, target_query, target_gallery, k=k_list, dim=0, return_index=return_index)
        return pred_index.transpose(0,1), correct.transpose(0,1)


def jaccard(a_list,b_list):
    return 1.0 - float(len(set(a_list)&set(b_list)))/float(len(set(a_list)|set(b_list)))*1.0
def topk(sim, target_gallery, target_query, k=[1,5,10], dim=1, print_index=False, reid_sim = None, return_index = False):
    result = []
    maxk = max(k)
    size_total = len(target_query)
    if reid_sim is None:
        _, pred_index = sim.topk(maxk, dim, True, True)
        pred_labels = target_gallery[pred_index]
    else:
        K = 5
        sim = sim.cpu().numpy()
        reid_sim = reid_sim.cpu().numpy()
        pred_index = np.argsort(-sim, axis = 1)
        reid_pred_index = np.argsort(-reid_sim, axis = 1)

        q_knn = pred_index[:, 0:K]
        g_knn = reid_pred_index[:, 0:K]

        new_index = []
        jaccard_dist = np.zeros_like(sim)
        from scipy.spatial import distance
        for i, qq in enumerate(q_knn):
            for j, gg in enumerate(g_knn):
                jaccard_dist[i, j] = 1.0 - jaccard(qq, gg)
        _, pred_index = torch.Tensor(sim + jaccard_dist).topk(maxk, dim, True, True)
        pred_labels = target_gallery[pred_index]
  

    # pred
    if dim == 1:
        pred_labels = pred_labels.t()

    correct = pred_labels.eq(target_query.view(1,-1).expand_as(pred_labels))

    if return_index == True:
        return pred_index, correct
    else:
        for topk in k:
            #correct_k = torch.sum(correct[:topk]).float()
            correct_k = torch.sum(correct[:topk], dim=0)
            correct_k = torch.sum(correct_k > 0).float()
            result.append(correct_k * 100 / size_total)
        return result

