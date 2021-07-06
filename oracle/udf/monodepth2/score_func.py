from __future__ import absolute_import, division, print_function

import sys
import os
import io
import glob

from oracle.udf.monodepth2 import test_simple
from oracle.udf.base import BaseScoringUDF
import config
sys.path.insert(0, os.path.dirname(__file__))

import torch
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from PIL import Image
from torchvision import transforms, datasets

from utils.parse_config import parse_model_config
from models import *
from utils.utils import *
import cv2
import numpy as np
import networks
from layers import disp_to_depth
from oracle.udf.monodepth2.utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR

matplotlib.use('agg')

obj_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

class Monodepth2(BaseScoringUDF):
    def __init__(self):
        super().__init__()
        self.arg_parser.add_argument("--class_thres", type=float, default=0.5)
        self.arg_parser.add_argument("--obj_thres", type=float, default=0)
        self.arg_parser.add_argument("--obj", type=str, choices=obj_names, default="car")
        self.model_config = 'config/yolov3.cfg'
        self.weights = 'weights/yolov3.weights'
    
    def initialize(self, opt, gpu=None):
        self.opt = opt
        self.obj = obj_names.index(opt.obj)
        self.device = config.device
        if gpu is not None:
            self.device = torch.device(f"cuda:{gpu}")
        self.model = Darknet(parse_model_config(self.model_config)).to(self.device)
        self.model.load_darknet_weights(self.weights)
        self.model.eval()
    
    def get_img_size(self):
        return (416, 416)
    
    def get_scores(self, imgs, visualize=False):
        assert (imgs.shape[1], imgs.shape[2]) == self.get_img_size()
        model_imgs = torch.from_numpy(imgs).float().to(self.device)
        model_imgs = model_imgs.permute(0, 3, 1, 2).contiguous().div(255)

        with torch.no_grad():
            detections = self.model(model_imgs)
            detections = non_max_suppression(detections, 0.1, 0.45)
        scores = []
        visual_imgs = []
        count = 0
        for i, boxes in enumerate(detections):
            max_score = 0
            if boxes is None:
                scores.append(0)
                if visualize:
                    visual_imgs.append(imgs[i])
            else:
                relavant_boxes = [box for box in boxes if int(box[-1]) == self.obj and float(box[4]) >= self.opt.class_thres and float(box[5]) >= self.opt.obj_thres]
                #if visualize:
                visual_img = np.copy(imgs[i])
                visual_img = cv2.resize(visual_img, (739, 416))

                for x1, y1, x2, y2, conf, cls_conf, cls_pred in relavant_boxes:
                    x1 = int(x1.item() / 416 * 739)
                    x2 = int(x2.item() / 416 * 739)
                    y1 = int(y1.item())
                    y2 = int(y2.item())
                    
                    depth = test_simple.test_simple(model_name="mono+stereo_640x192", input_image=visual_img, 
                        x1=x1, x2=x2, y1=y1, y2=y2, pred_metric_depth=True, count=count)
                    depth = depth.transpose(1,0)
                    #print("depth:", depth)
                    
                    average_score = 0
                    for x in range(x1+1, x2-1):
                        for y in range(y1+1, y2-1):
                            #print(depth[int(x*640/739)][int(y*192/416)])
                            if x > 738:
                                x = 738
                            if y > 415:
                                y = 415
                            average_score += (100 - depth[int(x*640/739)][int(y*192/416)])

                    average_score /= (x2-x1) * (y2-y1)
                    count += 1
                    if max_score < average_score:
                        max_score = average_score

                    cv2.rectangle(visual_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                #scores mainly range from 80-90, to show the difference, for a score like AB.CDEFG,
                #  transform it to BC
                scores.append(round(max_score * 10 - 800))

            #if visual_imgs:
                visual_imgs.append(Image.fromarray(visual_img))
        if visualize:
            return scores, visual_imgs
        else:
            return scores

        
        
