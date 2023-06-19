
import argparse
import logging
import pprint

from collections import OrderedDict

import numpy as np
import torch

import src.resnet as resnet
import src.wide_resnet as wide_resnet
from src.data_manager_clustervec import (
    init_data,
    make_transforms
)

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--use-pred', action='store_true',
    help='whether to use a prediction head')
parser.add_argument(
    '--model-name', type=str,
    help='model architecture',
    default='resnet50',
    choices=[
        "resnet18",
        'resnet50',
        'resnet50w2',
        'resnet50w4',
        'wide_resnet28w2'
    ])
parser.add_argument(
    '--pretrained', type=str,
    help='path to pretrained model',
    default='')
parser.add_argument(
    '--split-seed', type=float,
    default=152,
    help='seed for labeled data-split')
parser.add_argument(
    '--device', type=str,
    default='cuda:0',
    help='device to run script on')
parser.add_argument(
    '--unlabeled-frac', type=float,
    default='0.9',
    help='fraction of training data unlabeled')
parser.add_argument(
    '--normalize', type=bool,
    default=True,
    help='whether to standardize images before feeding to nework')
parser.add_argument(
    '--root-path', type=str,
    default='/datasets/',
    help='root directory to data')
parser.add_argument(
    '--image-folder', type=str,
    default='imagenet_full_size/061417/',
    help='image directory inside root_path')
parser.add_argument(
    '--dataset-name', type=str,
    default='imagenet_fine_tune',
    help='name of dataset to evaluate on',
    choices=[
        'ImageFolder_fine_tune',
        'datasetcsv_fine_tune'
    ])
parser.add_argument(
    '--subset-path', type=str,
    default='imagenet_subsets/',
    help='path to the csv folder'
   )
parser.add_argument(
    '--num_workers', type=str,
    help='Number of workers used by the dataloaders'
   )
