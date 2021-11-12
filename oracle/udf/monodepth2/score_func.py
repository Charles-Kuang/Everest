'''
python3 optical_flow.py --inference --model FlowNet2 --save_flow \
--inference_visualize \
--inference_dataset ImagesFromFolder \
--inference_dataset_root ./test \
--resume FlowNet2.pth.tar \
--save ./output \
'''
from __future__ import absolute_import, division, print_function

import sys
import os
import io
import glob

import subprocess

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

from pathlib import Path

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
    
    def get_scores(self, imgs, frames, visualize=False):
        #generate optical flow file
        if not os.path.exists('./oracle/udf/flownet2/output/000001.png'):
            subprocess.call("ffmpeg -i /mnt/everest/videos/Car_cam.mp4 \
             /mnt/everest/oracle/udf/flownet2/output/%06d.png", shell=True)
        if not os.path.exists('./oracle/udf/flownet2/output/000000.flo'):
            optical_flow = os.system("python3 ./oracle/udf/flownet2/optical_flow.py --inference --model FlowNet2 --save_flow \
            --inference_visualize \
            --inference_dataset ImagesFromFolder \
            --inference_dataset_root ./output \
            --resume ./FlowNet2.pth.tar \
            --save ./output")
            print("optical_flow:", optical_flow)

        assert (imgs.shape[1], imgs.shape[2]) == self.get_img_size()
        model_imgs = torch.from_numpy(imgs).float().to(self.device)
        model_imgs = model_imgs.permute(0, 3, 1, 2).contiguous().div(255)

        with torch.no_grad():
            detections = self.model(model_imgs)
            detections = non_max_suppression(detections, 0.1, 0.45)
        scores = []
        visual_imgs = []
        counts = 0
        for i, boxes in enumerate(detections):
            max_score = 0
            x1_m, x2_m, y1_m, y2_m = 0, 0, 0, 0
            if boxes is None:
                scores.append(0)
                if visualize:
                    visual_imgs.append(imgs[i])
            else:
                relavant_boxes = [box for box in boxes if int(box[-1]) == self.obj and float(box[4]) >= self.opt.class_thres and float(box[5]) >= self.opt.obj_thres]
                #if visualize:
                input_img = np.copy(imgs[i])

                ###
                output_directory = "oracle/udf/monodepth2/result/"
                image_path = "oracle/udf/monodepth2/result/" + str(counts) + "_disp.jpeg"
                output_name = os.path.splitext(os.path.basename(image_path))[0]
                with torch.no_grad():
                    #generate optical flow np.array, learned from https://cloud.tencent.com/developer/article/1461933
                    optical_flow_path = Path('oracle/udf/flownet2/output/' + ("%06d" % frames[i]) + '.flo')
                    with optical_flow_path.open(mode='r') as flo:
                        tag = np.fromfile(flo, np.float32, count=1)[0]
                        optical_width = np.fromfile(flo, np.int32, count=1)[0]
                        optical_height = np.fromfile(flo, np.int32, count=1)[0]
                        
                        #print('tag', tag, 'width', optical_width, 'height', optical_height)
                        
                        nbands = 2
                        tmp = np.fromfile(flo, np.float32, count= nbands * optical_width * optical_height)
                        flow = np.resize(tmp, (int(optical_height), int(optical_width), int(nbands)))

                    # PREDICTING ON EACH IMAGE IN TURN
                    # Load image and preprocess
                    processed_img = Image.fromarray(input_img,'RGB')

                    processed_img = processed_img.resize((self.feed_width, self.feed_height), Image.LANCZOS)
                    processed_img = transforms.ToTensor()(processed_img).unsqueeze(0)

                    # PREDICTION
                    processed_img = processed_img.to(device)
                    features = self.encoder(processed_img)
                    outputs = self.depth_decoder(features)

                    disp = outputs[("disp", 0)]
                    disp_resized = torch.nn.functional.interpolate(
                        disp, (480, 854), mode="bilinear", align_corners=False)

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

                depth = metric_depth[0][0]
                ###
                #print(self.feed_height, self.feed_width)

                depth = depth.transpose(1,0)
                
                height = input_img.shape[0]
                width = input_img.shape[1]
                input_img = cv2.resize(input_img, (854, 480))
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in relavant_boxes:
                    x1 = int(x1.item())
                    x2 = int(x2.item())
                    y1 = int(y1.item())
                    y2 = int(y2.item())
                    cv2.rectangle(input_img, (round(x1/width*854), round(y1/height*480)), (round(x2/width*854), round(y2/height*480)), (255, 0, 0), 2)

                    #filter out boxes whose center point is far from the lane
                    if round((x2+x1)/2) < round(width/4) or round((x2+x1)/2) > round(3*width/4):
                        continue

                    average_score = np.mean(100 - depth[round(x1/416*640):round(x2/416*640), round(y1/416*192):round(y2/416*192)])
                    counts += 1
                    if max_score < average_score:
                        max_score = average_score
                        x1_m, x2_m, y1_m, y2_m = x1, x2, y1, y2

                #get central block
                colormapped_im = cv2.resize(colormapped_im, (854, 480))
                cv2.rectangle(colormapped_im, (round(x1_m/416*854), round(y1_m/416*480)), (round(x2_m/416*854), round(y2_m/416*480)), (255, 0, 0), 2)
                #calculate optical flow score
                flow_x = np.mean(flow[round(x1_m/416*optical_width):round(x2_m/416*optical_width), round(y1_m/416*optical_height):round(y2_m/416*optical_height), 0])
                flow_y = np.mean(flow[round(x1_m/416*optical_width):round(x2_m/416*optical_width), round(y1_m/416*optical_height):round(y2_m/416*optical_height), 1])
                
                flow_x = abs(flow_x) + 1
                if(np.isnan(flow_x)):
                    max_score = max_score
                elif(((x1_m+x2_m)/2 < width / 2 and flow_x < 0) or ((x1_m+x2_m)/2 > width / 2 and flow_x > 0)):#moving away
                    max_score /= flow_x
                else:#approaching
                    max_score *= flow_x
                    

                flow_y = (abs(flow_y) + 1) * 2
                if(np.isnan(flow_y)):
                    max_score = max_score
                if(flow_y >= 0):#moving away
                    max_score /= flow_y
                else:#approaching
                    max_score *= flow_y
                
                if(np.isnan(max_score)):
                    max_score = 0

                
                flow_img_path = 'oracle/udf/flownet2/output/' + ("%06d" % frames[i]) + '-vis.png'
                #handle edge error
                if(frames[i] == 0):
                    flow_img_path = 'oracle/udf/flownet2/output/' + ("%06d" % (frames[i]+1)) + '-vis.png'
                else:
                    while(not os.path.isfile(flow_img_path)):
                        frames[i] -= 1
                        flow_img_path = 'oracle/udf/flownet2/output/' + ("%06d" % (frames[i])) + '-vis.png'

                flow_img = cv2.imread(flow_img_path)
                #print(flow_img.shape, optical_width, optical_height)
                cv2.rectangle(flow_img, (round(x1_m/416*optical_width), round(y1_m/416*optical_height)), (round(x2_m/416*optical_width), round(y2_m/416*optical_height)), (255, 0, 0), 2)
                flow_img = Image.fromarray(flow_img)
                flow_img.save('oracle/udf/flownet2/output/' + ("%06d" % frames[i]) + '-vis_rec.png')
                

                #scores range from 0-3000, to show the difference, for a score like AB.CDEFG,
                #  transform it to BC
                scores.append(max(round(np.sqrt(max_score)), 0))
                visual_img = cv2.vconcat([input_img, colormapped_im])

            #if visual_imgs:
                visual_imgs.append(Image.fromarray(visual_img))
        #print(scores)
        if visualize:
            return scores, visual_imgs
        else:
            return scores

        
        
