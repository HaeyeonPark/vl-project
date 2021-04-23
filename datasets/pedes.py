import torch.utils.data as data
import numpy as np
import os
import pickle
import h5py
import json
from PIL import Image
from utils.directory import check_exists
from scipy.misc import imread, imresize
import datasets.preprocess as preprocess
import collections
import random

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class CuhkPedes(data.Dataset):
    '''
    Args:
        root (string): Base root directory of dataset where [split].pkl and [split].h5 exists
        split (string): 'train', 'val' or 'test'
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed vector. E.g, ''transform.RandomCrop'
        target_transform (callable, optional): A funciton/transform that tkes in the
            targt and transfomrs it.
    '''

    def __init__(self, image_root, anno_root, split, max_length, transform=None, target_transform=None, \
                 cap_transform=None, vocab_path='', min_word_count=0, rand_sample=False):

        self.image_root = image_root
        self.anno_root = anno_root
        self.max_length = max_length
        self.transform = transform
        self.target_transform = target_transform
        self.cap_transform = cap_transform
        self.split = split.lower()
        self.vocab_path = vocab_path
        self.min_word_count = min_word_count
        self.rand_sample = rand_sample
        if self.rand_sample == True:
            self.label_to_caption=collections.defaultdict(list)
            self.data_aug=2 # how many times per image

        if not check_exists(self.image_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               'Please follow the directions to generate datasets')

        print('Reading data from json')
        data = self.get_data_from_json(split)
        self.read_data(data)

    
    def add_caption_to_data(self, split_data, data):

        fname_dict = {jj['file_path']:jj for jj in split_data}
        judge = {jj['file_path']: 0 for jj in split_data}

        for fname in data['images_path']:
            judge[fname] = judge[fname] + 1
        
        caption = []
        for fname in data['images_path']:
            jj = fname_dict[fname]

            caption.append(jj['captions'][judge[fname] - 1])
            judge[fname] = judge[fname] - 1
            
            
        assert len(caption) == len(data['images_path'])
        data['captions'] = caption
        return data
    
    def add_captions_to_data(self, split_data, data):
        captions=[]
        for el in split_data:
            captions.append(el['captions'])
        data['captions'] = captions

        return data

    def aggregate_captions(self, split_data, data):
        captions = []
        idx=0
        for el in split_data:
            label = el['id'] -1 # data['id]: 0~ split_data['id]: 1~ 
            for caption in el['captions']:
                captions.append(caption)
                self.label_to_caption[label].append(idx)
                idx+=1
        data['captions'] = captions
        return data

    def get_data_from_json(self, split):
        args = Namespace(min_word_count=self.min_word_count, remove_stopwords = None, out_root=None)

        split_data = self.load_split(self.split)
       
        if self.vocab_path == '':
            print('Building vocabulary...')
            vocab = preprocess.build_vocab(split_data, args, write=False)
        else:
            print('Loading vocabulary from {}'.format(self.vocab_path))
            vocab = self.load_vocab(self.vocab_path)

       
        split_metadata = preprocess.process_metadata(self.split, split_data, args, write=False)
        split_decodedata = preprocess.process_decodedata(split_metadata, vocab)
        data = preprocess.process_dataset(self.split, split_decodedata, args, write=False)
        

        #data = self.add_captions_to_data(split_data, data)
        #data = self.add_caption_to_data(split_data, data)
        if split =='train' and self.rand_sample==True:
            data = self.aggregate_captions(split_data, data)
        else:
            data = self.add_captions_to_data(split_data, data)
        return data

    def load_split(self, split):   
        split_root = os.path.join(self.anno_root, split + '_reid.json')
        with open(split_root, 'r') as f:
            split_data = json.load(f)        
        print('load {} data from json done'.format(split))
        
        return split_data

    def load_vocab(self, vocab_path):    
        with open(os.path.join(vocab_path), 'rb') as f:
            word_to_idx = pickle.load(f)

        vocab = preprocess.Vocabulary(word_to_idx, len(word_to_idx))
        print('load vocabulary done')
        return vocab   

    def read_data(self, data):
        
        if self.split == 'train':
            self.train_labels = data['labels']
            self.train_captions = data['captions']         
            self.train_images = data['images_path']

        elif self.split == 'val':
            self.val_labels = data['labels']
            self.val_captions = data['captions']
            self.val_images = data['images_path']

        elif self.split == 'test':
            self.test_labels = data['labels']
            self.test_captions = data['captions']
            self.test_images = data['images_path']

            unique = []
            new_test_images = []
            for test_image in self.test_images:
                if test_image in new_test_images:
                    unique.append(0)
                else:
                    unique.append(1)
                    new_test_images.append(test_image)
            self.unique = unique

        
        else:
            raise RuntimeError('Wrong split which should be one of "train","val" or "test"')


    def __getitem__(self, index):
        """
        Args:
              index(int): Index
        Returns:
              tuple: (images, labels, captions)
        """
        if self.split == 'train':
            if self.rand_sample == True:
                num_images = len(self.train_labels) # 34054
                label, img_path = self.train_labels[index % num_images], self.train_images[index % num_images]
                # randomly sample 2 captions per image and aggregate to list  
                #assert len(self.label_to_caption[label])>=2
                caption_idx = random.sample(self.label_to_caption[label],2)
                caption = [self.train_captions[cidx] for cidx in caption_idx]
            else:
                img_path, caption, label = self.train_images[index], self.train_captions[index], self.train_labels[index]
        elif self.split == 'val':
            img_path, caption, label = self.val_images[index], self.val_captions[index], self.val_labels[index]
        else:
            img_path, caption, label = self.test_images[index], self.test_captions[index], self.test_labels[index]

        #image
        middle_path = "CUHK-PEDES/imgs"
        if middle_path not in img_path:
            img_path = os.path.join(self.image_root, middle_path, img_path)
        else:
            img_path = os.path.join(self.image_root, img_path)


        img = imread(img_path)
        img = imresize(img, (384,128))
        if len(img.shape) == 2:
            img = np.dstack((img,img,img))
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)
        
        # caption 
        
        if self.split == 'train':
            return img, caption, label
        else:
            return img, caption, label                        

    def __len__(self):
        if self.split == 'train':
            if self.rand_sample == True:
                return len(self.train_labels) * self.data_aug
            else:
                return len(self.train_labels)
        elif self.split == 'val':
            return len(self.val_labels)
        else:
            return len(self.test_labels)
