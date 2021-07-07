from __future__ import absolute_import, division, print_function

import sys
import os
import io
import glob

from oracle.udf.base import BaseScoringUDF
import config
sys.path.insert(0, os.path.dirname(__file__))

import torch
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from PIL import Image
import PIL.ImageDraw as pil_draw
from torchvision import transforms, datasets

from utils.parse_config import parse_model_config
from models import *
from utils.utils import *
import cv2
import numpy as np
import networks
from layers import disp_to_depth
from oracle.udf.monodepth2.utils import download_model_if_doesnt_exist

STEREO_SCALE_FACTOR = 5.4

mpl.use('agg')

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

        model_name="mono+stereo_640x192"
        download_model_if_doesnt_exist(model_name)
        model_path = os.path.join("oracle/udf/monodepth2/models", model_name)
        #print("-> Loading model from ", model_path)
        encoder_path = os.path.join(model_path, "encoder.pth")
        depth_decoder_path = os.path.join(model_path, "depth.pth")

        # LOADING PRETRAINED MODEL
        #print("   Loading pretrained encoder")
        self.encoder = networks.ResnetEncoder(18, False)
        loaded_dict_enc = torch.load(encoder_path, map_location=device)

        # extract the height and width of image that this model was trained with
        self.feed_height = loaded_dict_enc['height']
        self.feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)
        self.encoder.to(device)
        self.encoder.eval()

        #print("   Loading pretrained decoder")
        self.depth_decoder = networks.DepthDecoder(
            num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        loaded_dict = torch.load(depth_decoder_path, map_location=device)
        self.depth_decoder.load_state_dict(loaded_dict)

        self.depth_decoder.to(device)
        self.depth_decoder.eval()

        
    
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
        pic_num = -1
        for i, boxes in enumerate(detections):
            pic_num += 1
            max_score = 0
            if boxes is None:
                scores.append(0)
                if visualize:
                    visual_imgs.append(imgs[i])
            else:
                relavant_boxes = [box for box in boxes if int(box[-1]) == self.obj and float(box[4]) >= self.opt.class_thres and float(box[5]) >= self.opt.obj_thres]
                #if visualize:
                visual_img = np.copy(imgs[i])

                for x1, y1, x2, y2, conf, cls_conf, cls_pred in relavant_boxes:
                    x1 = int(x1.item())
                    x2 = int(x2.item())
                    y1 = int(y1.item())
                    y2 = int(y2.item())

                    #filter out boxes whose center point is far from the lane
                    height = visual_img.shape[0]
                    width = visual_img.shape[1]
                    if not (round((x2+x1)/2) in range(round(width/4), round(3*width/4))):
                        continue
                    
                    ###
                    output_directory = "oracle/udf/monodepth2/result/"
                    image_path = "oracle/udf/monodepth2/result/" + str(count) + "_disp.jpeg"
                    output_name = os.path.splitext(os.path.basename(image_path))[0]
                    with torch.no_grad():
                        # PREDICTING ON EACH IMAGE IN TURN
                        # Load image and preprocess
                        input_img = Image.fromarray(visual_img,'RGB')
                        im1 = input_img
                        name_dest_im1 = os.path.join(output_directory, "{}_disp1.jpeg".format(output_name))
                        name_dest_im2 = os.path.join(output_directory, "{}_disp2.jpeg".format(output_name))
                        #imd = pil_draw.Draw(im1)
                        #imd.rectangle([(x1, y1), (x2, y2)],  outline="red")
                        #im1.save(name_dest_im1)
                        #imd.save(name_dest_im2)

                        original_width, original_height = input_img.size
                        input_img = input_img.resize((self.feed_width, self.feed_height), Image.LANCZOS)
                        input_img = transforms.ToTensor()(input_img).unsqueeze(0)

                        # PREDICTION
                        input_img = input_img.to(device)
                        features = self.encoder(input_img)
                        outputs = self.depth_decoder(features)

                        disp = outputs[("disp", 0)]
                        disp_resized = torch.nn.functional.interpolate(
                            disp, (original_height, original_width), mode="bilinear", align_corners=False)

                        # Saving numpy file
                        scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
                        
                        name_dest_npy = os.path.join(output_directory, "{}_depth.npy".format(output_name))
                        metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()
                        #np.save(name_dest_npy, metric_depth)

                        # Saving colormapped depth image
                        disp_resized_np = disp_resized.squeeze().cpu().numpy()
                        vmax = np.percentile(disp_resized_np, 95)
                        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
                        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
                        im = Image.fromarray(colormapped_im)

                        name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))

                        
                        #imt = pil_draw.Draw(im)
                        #imt.rectangle([(x1, y1), (x2, y2)], outline="red")
                        #im.save(name_dest_im)

                        #print('-> Done!')
                    return_val = STEREO_SCALE_FACTOR * scaled_disp.cpu().numpy()#scaled_disp
                    depth = metric_depth[0][0]
                    ###
                    #print(self.feed_height, self.feed_width)

                    depth = depth.transpose(1,0)
                    average_score = np.mean(100 - depth[round(x1/416*640):round(x2/416*640), round(y1/416*192):round(y2/416*192)])
                    count += 1
                    if max_score < average_score:
                        max_score = average_score

                    cv2.rectangle(visual_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                #scores mainly range from 80-90, to show the difference, for a score like AB.CDEFG,
                #  transform it to BC
                scores.append(max(round(max_score * 10 - 800), 0))

            #if visual_imgs:
                visual_imgs.append(Image.fromarray(visual_img))
        if visualize:
            return scores, visual_imgs
        else:
            return scores

        
        
