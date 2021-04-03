import argparse
from config import log_config 
import logging


def parse_args():
    parser = argparse.ArgumentParser(description='command for evaluate on CUHK-PEDES')
    # Directory
    parser.add_argument('--image_dir', type=str, default='/workspace/data', help='directory to store dataset')
    parser.add_argument('--anno_dir', type=str, default='/workspace/code/data/processed_data',help='directory to store anno file')
    parser.add_argument('--model_path', type=str, default ='/workspace/code/model_data/exp5',help='directory to exp ')
    parser.add_argument('--log_dir', type=str, default='/workspace/code/logs/exp5', help='directory to store log')

    parser.add_argument('--feature_size', type=int, default=768)
    parser.add_argument('--cnn_dropout_keep', type=float, default=0.999)
    parser.add_argument('--part2', type=int, default=2, help='number of stripes splited in patch branch')
    parser.add_argument('--part3', type=int, default=3, help='number of stripes splited in region branch')
    parser.add_argument('--focal_type', type=str, default=None)
    parser.add_argument('--lambda_softmax', type=float, default=20.0, help='scale constant')
    parser.add_argument('--reranking', action='store_true', default=True, help='whether reranking during testing')
    
    # Default setting
    parser.add_argument('--gpus', type=str, default='0,1')
    parser.add_argument('--epoch_start', type=int, default=0)
    parser.add_argument('--checkpoint_dir', type=str, default='')


    args = parser.parse_args()
    return args



def config():
    args = parse_args()
    log_config(args, 'test')
    return args
