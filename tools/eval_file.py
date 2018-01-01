import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
from datasets import coco
import caffe
import argparse
import pprint
import time, os, sys


if __name__=='__main__':

    coco_set = coco.coco('val','2014')
    res_file = 'output/1018b/coco_2014_val/res101_rfcn_coco_1018b_64w_iter_320000/detections_val2014_results_656b2abb-5d22-4240-80d8-3f884133eef8.json'
    output_dir = 'output/1018b/coco_2014_val/res101_rfcn_coco_1018b_64w_iter_320000'
    coco_set._do_detection_eval(res_file, output_dir)